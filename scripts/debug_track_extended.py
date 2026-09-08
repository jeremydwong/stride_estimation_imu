#!/usr/bin/env python3
"""Extended headless track investigation: mechanization, controls, and validation.

Run after/alongside debug_track_foot_difference.py. All data are read locally.
The selected candidate does NOT constrain XY, distance, heading or loop closure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import debug_track_foot_difference as base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import h5py
from types import SimpleNamespace
import stride_imu as imu
from stride_imu.inertial import foot_fall, zero_velocity_updates, GRAVITY
from stride_imu.experimental import compute_position_experimental, compute_position_stride_gravity
from track_research_models import eskf, synthetic


def stable_calibration(W,A,period,window_s=.5,gyro_std_deg_s=.3,accel_std=.05):
    """Sensitivity control: only low-variance windows of the separate quiet bout.

    Movement-free calibration cannot be guaranteed from these thresholds. This
    seemingly cleaner estimate was NOT selected just for a lower track RMSE.
    """
    count=int(round(window_s/period));accepted=[]
    for start in range(0,len(W)-count+1,count):
        w=W[start:start+count]/period;a=A[start:start+count]
        if (np.linalg.norm(np.std(w,axis=0))<np.deg2rad(gyro_std_deg_s)
            and np.linalg.norm(np.std(a,axis=0))<accel_std
            and np.linalg.norm(np.mean(w,axis=0))<np.deg2rad(2)):
            accepted.extend(range(start,start+count))
    if len(accepted)*period<1:
        raise ValueError('Fewer than 1 s of accepted stationary calibration data')
    return np.mean(W[accepted],axis=0)/period,len(accepted)*period


def vendor_orientation(data,metadata,side):
    """APDM-provided attitude control; not a ground-truth heading reference.

    Scalar-first quaternion/body-to-nav convention checked against quiet gravity.
    Apply to original sensor acceleration, not the already transformed body axes.
    APDM gives +Z gravity here; rotate 180 degrees about X to package -Z gravity.
    """
    entry=metadata['selected'][side]
    with h5py.File(base.ROOT/'data'/base.FILES[side]) as f:
        sensor=list(f['Sensors'])[0];s,e=entry['start_idx'],entry['end_idx']
        q=f['Processed'][sensor]['Orientation'][s:e]
        a=f['Sensors'][sensor]['Accelerometer'][s:e]
    an=Rotation.from_quat(q[:,[1,2,3,0]]).apply(a)
    ff,stationary=foot_fall(data[side+'_W'],data[side+'_A'],float(data['period']))
    if np.median(an[stationary,2])<0:
        raise ValueError('Unexpected APDM gravity convention')
    an[:,1:]*=-1
    anz=np.zeros_like(an);last=0
    for i in range(1,len(an)):last=zero_velocity_updates(i,ff,an,anz,last)
    p=np.cumsum(np.cumsum(anz,axis=0)*float(data['period']),axis=0)*float(data['period'])
    return SimpleNamespace(P=p,FF=ff)


def extended_configs():
    return [
        ('baseline','old',{}), ('v1_gravity_bias','sample',{}),
        ('sample_init','sample',dict(initial=True)),
        ('sample_post','sample',dict(rotate_after_tilt=True)),
        ('sample_trapezoid','sample',dict(integration='trapezoid')),
        ('sample_init_trap_post','sample',dict(initial=True,integration='trapezoid',rotate_after_tilt=True)),
        ('sample_init_trap_midbias','sample',dict(initial=True,integration='trapezoid',rotate_after_tilt=True,bias='mid')),
        ('sample_init_trap_meanbias','sample',dict(initial=True,integration='trapezoid',rotate_after_tilt=True,bias='mean')),
        ('stride_gravity','stride',{}),
        ('stride_after_bias','stride',dict(bias='after')),
        ('stride_mid_bias','stride',dict(bias='mid')),
        ('stride_trapezoid','stride',dict(integration='trapezoid')),
        ('stride_trap_midbias','stride',dict(integration='trapezoid',bias='mid')),
        ('stride_gain_half','stride',dict(gravity_gain=.5)),
        ('stride_gain_tenth','stride',dict(gravity_gain=.1)),
        ('stride_flat_prior','stride',dict(assume_level_contacts=True)),
        ('stride_no_bias','stride',dict(bias='none')),
        ('stride_stable_cal','stride',dict(bias='stable')),
        ('eskf_sparse','eskf',{}), ('eskf_dense','eskf',dict(dense=True)),
        ('eskf_sparse_floor','eskf',dict(floor=True)),
        ('eskf_dense_floor','eskf',dict(dense=True,floor=True)),
        ('eskf_fixed_bias','eskf',dict(bias_sigma=0,acc_sigma=0)),
        ('eskf_dense_fixed','eskf',dict(dense=True,bias_sigma=0,acc_sigma=0)),
        ('eskf_bias1','eskf',dict(bias_sigma=1)),
        ('eskf_floor_bias1','eskf',dict(floor=True,bias_sigma=1)),
        ('eskf_acc02','eskf',dict(acc_sigma=.2)),
        ('eskf_floor_acc02','eskf',dict(floor=True,acc_sigma=.2)),
        ('eskf_dense_sigma01','eskf',dict(dense=True,zupt_sigma=.1)),
        ('eskf_right','eskf',dict(trap=False)),
        ('APDM_orientation','vendor',{}),
    ]


def run_real(data,metadata,output,configs=None):
    configs=extended_configs() if configs is None else configs
    rows=[];selected={};dt=float(data['period'])
    for name,model,config in configs:
        pair={}
        for side in base.FILES:
            kw=dict(config);bias_method=kw.pop('bias','before')
            w,a=data[side+'_W'],data[side+'_A']
            before=data[side+'_bias_before'];after=data[side+'_bias_after']
            bias={'before':before,'after':after,'mid':(before+after)/2,'none':np.zeros(3),
                  'mean':np.mean(data[side+'_quiet_before_W'],axis=0)/dt}.get(bias_method)
            if bias_method=='stable':
                bias,_=stable_calibration(data[side+'_quiet_before_W'],data[side+'_quiet_before_A'],dt)
            a0=np.median(data[side+'_quiet_before_A'],axis=0)
            if model=='old':r=imu.compute_position(w,a,dt)
            elif model=='sample':
                if kw.pop('initial',False):kw['initial_gravity']=a0
                r=compute_position_experimental(w,a,dt,gyro_bias_rad_s=bias,**kw)
            elif model=='stride':r=compute_position_stride_gravity(w,a,dt,gyro_bias_rad_s=bias,initial_gravity=a0,**kw)
            elif model=='eskf':r=eskf(w,a,dt,bias,a0,**kw)
            elif model=='vendor':r=vendor_orientation(data,metadata,side)
            else:raise ValueError(model)
            pair[side]=r
            rows.append(dict(experiment=name,foot=side,
                stance_distance_m=np.linalg.norm(np.diff(r.P[r.FF,:2],axis=0),axis=1).sum(),
                endpoint_gap_m=np.linalg.norm(r.P[-1,:2]-r.P[0,:2]),
                elevation_range_m=np.ptp(r.P[:,2]),end_elevation_m=r.P[-1,2]-r.P[0,2],
                contacts=int(r.FF.sum())))
        rmse,p95=base.score_pair(pair,data)
        for row in rows[-2:]:row.update(pair_rmse_m=rmse,pair_p95_m=p95)
        if name in ['baseline','v1_gravity_bias','stride_gravity','stride_flat_prior']:
            selected[name]=pair
        print(f'{name:28s} XY pair {rmse:6.3f} m; Z ranges {rows[-2]["elevation_range_m"]:6.3f}/{rows[-1]["elevation_range_m"]:6.3f} m',flush=True)
    metrics=pd.DataFrame(rows);metrics.to_csv(Path(output)/'extended_metrics.csv',index=False)
    return selected,metrics


def run_synthetic(output):
    rows=[];plots={};gate_rows=[]
    for noise in [False,True]:
        for side in base.FILES:
            t,w,a,truth,bias,a0=synthetic(side)
            if noise:
                rng=np.random.default_rng(123 if side=='left' else 124)
                w+=rng.normal(0,np.deg2rad(.05)/128,w.shape)
                a+=rng.normal(0,.03,a.shape)
            velocity=np.gradient(truth,1/128,axis=0)
            acceleration=np.gradient(velocity,1/128,axis=0)
            gate=(np.rad2deg(np.linalg.norm(w,axis=1)*128)<30)&(np.abs(np.linalg.norm(a,axis=1)-GRAVITY)<1)
            gate_rows.append(dict(foot=side,noise=noise,tilt_gate_samples=int(sum(gate)),
                accepted_accelerating_samples=int(sum(gate&(np.linalg.norm(acceleration,axis=1)>.5)))))
            tp,_=base.align_initial(truth,t)
            for name in ['baseline','v1_gravity_bias','stride_gravity']:
                if name=='baseline':r=imu.compute_position(w,a,1/128)
                elif name=='v1_gravity_bias':r=compute_position_experimental(w,a,1/128,gyro_bias_rad_s=bias)
                else:r=compute_position_stride_gravity(w,a,1/128,gyro_bias_rad_s=bias,initial_gravity=a0)
                p,_=base.align_initial(r.P,t)
                rows.append(dict(experiment=name,foot=side,noise=noise,
                    true_xy_rmse_m=np.sqrt(np.mean(np.sum((p-tp)**2,axis=1))),
                    endpoint_gap_m=np.linalg.norm(p[-1]),elevation_range_m=np.ptp(r.P[:,2]),
                    stance_distance_m=np.linalg.norm(np.diff(r.P[r.FF,:2],axis=0),axis=1).sum()))
                if not noise:plots[(name,side)]=(p,tp)
    frame=pd.DataFrame(rows);frame.to_csv(Path(output)/'synthetic_metrics.csv',index=False)
    gates=pd.DataFrame(gate_rows);gates.to_csv(Path(output)/'synthetic_gate_audit.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,5),sharex=True,sharey=True)
    for ax,name in zip(axes,['baseline','v1_gravity_bias','stride_gravity']):
        for side in base.FILES:
            p,tp=plots[(name,side)];ax.plot(*p.T,label=side)
        ax.plot(*tp.T,'k--',label='known path',lw=1)
        ax.set_title(name);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.3);ax.legend()
    fig.suptitle('Known 200 m synthetic stadium, 12 m corner radius, asymmetric foot swings')
    fig.tight_layout();fig.savefig(Path(output)/'synthetic.png',dpi=150);plt.close(fig)
    return frame,gates


def calibration_sweep(data, output):
    rows=[];dt=float(data['period'])
    for window_s in [.25,.5,1.]:
        for threshold in [.15,.3,.5]:
            pair={};row=dict(window_s=window_s,gyro_std_deg_s=threshold)
            try:
                for side in base.FILES:
                    bias,seconds=stable_calibration(data[side+'_quiet_before_W'],
                        data[side+'_quiet_before_A'],dt,window_s,threshold)
                    row[side+'_accepted_s']=seconds
                    pair[side]=compute_position_stride_gravity(data[side+'_W'],data[side+'_A'],dt,
                        gyro_bias_rad_s=bias,initial_gravity=np.median(data[side+'_quiet_before_A'],axis=0))
                row['pair_rmse_m']=base.score_pair(pair,data)[0]
                row['status']='ok'
            except ValueError as error:
                row['status']=str(error)
            rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(Path(output)/'calibration_sweep.csv',index=False)
    return frame


def plot_extended(selected,data,output,title='Track bout'):
    names=['baseline','v1_gravity_bias','stride_gravity']
    fig,axes=plt.subplots(2,3,figsize=(14,9))
    for col,name in enumerate(names):
        for side,r in selected[name].items():
            p,_=base.align_initial(r.P,data[side+'_t'])
            axes[0,col].plot(*p.T,label=side)
            axes[1,col].plot(data[side+'_t'],r.P[:,2],label=side)
        axes[0,col].set_title(f'{name}\nXY disagreement {base.score_pair(selected[name],data)[0]:.2f} m')
        axes[0,col].set_aspect('equal',adjustable='box');axes[0,col].set_xlabel('X [m]');axes[0,col].set_ylabel('Y [m]')
        axes[1,col].set_xlabel('Time [s]');axes[1,col].set_ylabel('Unconstrained Z [m]')
        for row in range(2):axes[row,col].grid(alpha=.3);axes[row,col].legend()
    # Same vertical scale makes the reduction reviewable.
    limits=[ax.get_ylim() for ax in axes[1]]
    for ax in axes[1]:ax.set_ylim(min(x[0] for x in limits),max(x[1] for x in limits))
    fig.suptitle(title+' — no floor, heading, circumference or closure constraint')
    fig.tight_layout();fig.savefig(Path(output)/'extended_comparison.png',dpi=150);plt.close(fig)


def run_all(output=base.DEFAULT_OUTPUT):
    output=Path(output)
    data,metadata=base.extract_track_bout(output)
    selected,metrics=run_real(data,metadata,output)
    plot_extended(selected,data,output)
    validation_output=output/'validation'
    validation,validation_metadata=base.extract_track_bout(validation_output,bout_rank=1)
    configs=[c for c in extended_configs() if c[0] in ['baseline','v1_gravity_bias','stride_gravity']]
    validation_selected,validation_metrics=run_real(validation,validation_metadata,validation_output,configs)
    plot_extended(validation_selected,validation,validation_output,'Separate 16:31 walking bout (geometry unconfirmed)')
    synthetic_metrics,gates=run_synthetic(output)
    calibration_sweep(data,output)
    print(synthetic_metrics.round(3).to_string(index=False))
    return data,selected,metrics,validation_metrics,synthetic_metrics,gates


if __name__=='__main__':run_all()
