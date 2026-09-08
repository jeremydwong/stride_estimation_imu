#!/usr/bin/env python3
"""Reproduce the single-foot demo's track selection and run headless ablations.

Run: .venv/bin/python scripts/debug_track_foot_difference.py
Outputs are regenerated under debug/track_foot_difference; the companion
notebook imports these helpers and embeds the figures and interpretation.
"""
from pathlib import Path
import os
import sys
import json
import argparse
from datetime import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/stride-track-mpl')
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stride_imu as imu
from stride_imu.inertial import GRAVITY
from stride_imu.experimental import compute_position_experimental

FILES = {'left': '20251029-154305_LF_Pilot_Ch_Oct29.h5',
         'right': '20251029-154310_RF_Pilot_Ch_Oct29.h5'}
DEFAULT_OUTPUT = ROOT / 'debug' / 'track_foot_difference'


def extract_track_bout(output=DEFAULT_OUTPUT):
    """Copy demo detection/longest-near-16:34 selection, using BOTH actual files.

    Select independently to audit detection, then use the union of both selected
    time intervals. Preserve sensor timestamps and interpolate only for scoring;
    never pair equal sample indices from independently started recordings.
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    records = {side: imu.load_imu_recording(str(ROOT/'data'/file)) for side, file in FILES.items()}
    chosen, metadata = {}, {'files': FILES, 'target_local': '16:34:00', 'search_seconds': 240}
    candidates = {}
    for side, rec in records.items():
        bouts = imu.detect_walking_bouts(rec.Wb, rec.Ab, rec.period,
                  W_threshold=30., A_threshold=1., min_quiet_seconds=1.5, min_walk_seconds=3.)
        matches = imu.find_bouts_near_time(bouts, rec.time_datetime, time(16,34), 240)
        if not matches:
            raise ValueError(f'No track candidates for {side}')
        chosen[side] = max(matches, key=lambda b: b.duration_seconds)
        candidates[side] = [{'start': rec.time_datetime[b.start_idx].isoformat(),
                             'duration_s': b.duration_seconds} for b in matches]
    start = min(rec.raw_time[chosen[s].start_idx] for s, rec in records.items())
    end = max(rec.raw_time[chosen[s].end_idx-1] for s, rec in records.items())
    period = records['left'].period
    if not np.isclose(period, records['right'].period):
        raise ValueError('Different sampling periods; this experiment requires matching rates')
    arrays = {'period': period}
    metadata.update(candidates=candidates, selected={}, calibration={})
    for side, rec in records.items():
        b = chosen[side]
        idx = np.flatnonzero((rec.raw_time >= start) & (rec.raw_time <= end))
        s, e = int(idx[0]), int(idx[-1]+1)
        arrays[side+'_W'] = rec.Wb[s:e]
        arrays[side+'_A'] = rec.Ab[s:e]
        arrays[side+'_t'] = (rec.raw_time[s:e].astype(float)-float(start))/1e6
        # Separate stationary intervals, not moving stance samples. Keep the
        # complete detector-selected intervals, matching the initial experiment.
        before = rec.Wb[b.quiet_before_idx:b.start_idx]
        after = rec.Wb[b.end_idx:b.quiet_after_idx]
        arrays[side+'_bias_before'] = np.median(before, axis=0)/period
        arrays[side+'_bias_after'] = np.median(after, axis=0)/period
        metadata['selected'][side] = dict(start=rec.time_datetime[s].isoformat(),
            end=rec.time_datetime[e-1].isoformat(), start_idx=s, end_idx=e, samples=e-s,
            independent_duration_s=b.duration_seconds)
        metadata['calibration'][side] = dict(before_seconds=len(before)*period,
            after_seconds=len(after)*period,
            before_bias_deg_s=np.rad2deg(arrays[side+'_bias_before']).tolist(),
            after_bias_deg_s=np.rad2deg(arrays[side+'_bias_after']).tolist())
    np.savez_compressed(output/'bout.npz', **arrays)
    (output/'bout_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return arrays, metadata


def align_initial(P, t, seconds=10.):
    """Translation + one rotation from initial travel, never scaling/reflection.

    Unlike the demo's endpoint alignment, this stays defined near loop closure.
    Each sensor has an arbitrary initial heading; compare initial walking travel.
    """
    p = P[:,:2] - P[0,:2]
    direction = np.array([np.interp(seconds, t, p[:, j]) for j in range(2)])
    if np.linalg.norm(direction) < 1.:
        raise ValueError('Initial displacement too small for heading alignment')
    angle = np.pi/2 - np.arctan2(direction[1], direction[0])
    c, s = np.cos(angle), np.sin(angle)
    return p @ np.array([[c, s], [-s, c]]), np.rad2deg(angle)


def experiment_configs():
    # One-factor ablations and a compact sensitivity sweep; not fitted to 200 m.
    return [
        ('baseline', None),
        ('legacy_bias', dict(tilt_method='legacy', bias='before')),
        ('gravity_only', dict(tilt_method='gravity')),
        ('gravity_bias', dict(tilt_method='gravity', bias='before')),
        ('gravity_bias_after', dict(tilt_method='gravity', bias='after')),
        ('tilt_off', dict(tilt_method='off')),
        ('legacy_tilt_w10', dict(tilt_method='legacy', tilt_w_deg_s=10.)),
        ('legacy_tilt_w60', dict(tilt_method='legacy', tilt_w_deg_s=60.)),
        ('legacy_q0.1', dict(tilt_method='legacy', tilt_process_noise=1e-6)),
        ('legacy_q10', dict(tilt_method='legacy', tilt_process_noise=1e-4)),
        ('footfall_w10', dict(tilt_method='legacy', W_FF=10., tilt_w_deg_s=10.)),
        ('footfall_w60', dict(tilt_method='legacy', W_FF=60., tilt_w_deg_s=60.)),
        ('gravity_bias_w15', dict(bias='before', tilt_w_deg_s=15.)),
        ('gravity_bias_w60', dict(bias='before', tilt_w_deg_s=60.)),
        ('gravity_bias_a0.5', dict(bias='before', tilt_a_m_s2=.5)),
        ('gravity_bias_a2', dict(bias='before', tilt_a_m_s2=2.)),
        ('gravity_bias_q0.1', dict(bias='before', tilt_process_noise=1e-6)),
        ('gravity_bias_q10', dict(bias='before', tilt_process_noise=1e-4)),
    ]


def score_pair(results, data, alignment_seconds=10.):
    grid = np.arange(max(data[s+'_t'][0] for s in FILES),
                     min(data[s+'_t'][-1] for s in FILES), .1)
    positions = []
    for side in FILES:
        p, _ = align_initial(results[side].P, data[side+'_t'], alignment_seconds)
        positions.append(np.column_stack([np.interp(grid, data[side+'_t'], p[:, j]) for j in range(2)]))
    distance = np.linalg.norm(positions[0]-positions[1], axis=1)
    return float(np.sqrt(np.mean(distance**2))), float(np.percentile(distance,95))


def run_experiments(data, output=DEFAULT_OUTPUT):
    period = float(data['period']); output = Path(output)
    results, rows = {}, []
    for name, config in experiment_configs():
        results[name] = {}
        for side in FILES:
            W, A = data[side+'_W'], data[side+'_A']
            kwargs = {} if config is None else dict(config)
            if 'bias' in kwargs:
                kwargs['gyro_bias_rad_s'] = data[side+'_bias_'+kwargs.pop('bias')]
            r = (imu.compute_position(W,A,period) if config is None else
                 compute_position_experimental(W,A,period,**kwargs))
            results[name][side] = r
            p, angle = align_initial(r.P,data[side+'_t'])
            tilt_mask = ((np.rad2deg(np.linalg.norm(W,axis=1)/period)<kwargs.get('tilt_w_deg_s',30.)) &
                         (np.abs(np.linalg.norm(A,axis=1)-GRAVITY)<kwargs.get('tilt_a_m_s2',1.)))
            rows.append(dict(experiment=name, foot=side,
                stance_distance_m=np.linalg.norm(np.diff(r.P[r.FF,:2],axis=0),axis=1).sum(),
                raw_3d_distance_m=np.linalg.norm(np.diff(r.P,axis=0),axis=1).sum(),
                endpoint_gap_m=np.linalg.norm(p[-1]), elevation_range_m=np.ptp(r.P[:,2]),
                contacts=int(r.FF.sum()), tilt_samples=int(tilt_mask.sum()) if kwargs.get('tilt_method')!='off' else 0,
                initial_alignment_deg=angle))
        rmse, p95 = score_pair(results[name],data)
        for row in rows[-2:]: row.update(pair_rmse_m=rmse,pair_p95_m=p95)
        print(f'{name:24s} pair RMSE {rmse:6.2f} m; contacts {rows[-2]["contacts"]}/{rows[-1]["contacts"]}',flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(output/'metrics.csv',index=False)
    # Cache small plotting outputs, not all accelerations from every sweep run.
    arrays = {f'{name}__{side}__{key}':getattr(r,key)
              for name,pair in results.items() for side,r in pair.items()
              for key in ('P','FF','euler')}
    np.savez_compressed(output/'trajectories.npz',**arrays)
    return results,frame


def plot_trajectory_comparison(results, data):
    names = ['baseline','legacy_bias','gravity_only','gravity_bias']
    fig, axes = plt.subplots(1,4,figsize=(16,5),sharex=True,sharey=True)
    for ax,name in zip(axes,names):
        for side,r in results[name].items():
            p,_ = align_initial(r.P,data[side+'_t'])
            ax.plot(*p.T,label=side,lw=1.1)
            ax.scatter(*p[-1],s=15)
        ax.set_title(f'{name}\nRMSE {score_pair(results[name],data)[0]:.2f} m')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.3);ax.set_xlabel('X [m]')
    axes[0].set_ylabel('Y [m]');axes[0].legend()
    fig.suptitle('Same time window; first 10 s heading alignment; no scaling or loop constraint')
    fig.tight_layout();return fig


def plot_alignment_artifact(results,data):
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for side,r in results['baseline'].items():
        for ax,seconds in zip(axes,[data[side+'_t'][-1],10.]):
            p,_=align_initial(r.P,data[side+'_t'],seconds)
            ax.plot(*p.T,label=side)
    for ax,title in zip(axes,['Demo endpoint alignment','Initial 10-second alignment']):
        ax.set_title(title);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.3)
        ax.set_xlabel('X [m]');ax.set_ylabel('Y [m]');ax.legend()
    fig.tight_layout();return fig


def plot_orientation(results,data):
    fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True)
    for col,side in enumerate(FILES):
        for name in ['baseline','gravity_bias']:
            r=results[name][side];t=data[side+'_t'];ff=np.flatnonzero(r.FF)
            axes[0,col].plot(t,np.rad2deg(r.euler[:,1]),alpha=.35,lw=.5,label=name)
            # Sample heading at contacts: unwrapping swing Euler yaw can count
            # spurious whole revolutions at pitch singularities.
            axes[1,col].plot(t[ff],np.rad2deg(np.unwrap(r.euler[ff,2])),label=name)
            p,_=align_initial(r.P,t);dp=np.diff(p[ff],axis=0)
            valid=np.linalg.norm(dp,axis=1)>.5
            heading=np.unwrap(np.arctan2(dp[valid,1],dp[valid,0]))
            axes[2,col].plot(t[ff[1:][valid]],np.rad2deg(heading),label=name)
        axes[0,col].set_title(side)
    for row,label in enumerate(['Swing pitch [deg]','Contact Euler yaw [deg]','Stride travel heading [deg]']):
        for ax in axes[row]:ax.set_ylabel(label);ax.grid(alpha=.3);ax.legend()
    for ax in axes[-1]:ax.set_xlabel('Seconds into common bout')
    fig.tight_layout();return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    data,metadata=extract_track_bout(args.output)
    results,metrics=run_experiments(data,args.output)
    for name,fn in [('trajectories',plot_trajectory_comparison),('alignment',plot_alignment_artifact),('orientation',plot_orientation)]:
        fig=fn(results,data);fig.savefig(args.output/f'{name}.png',dpi=150);plt.close(fig)
    print(metrics[metrics.experiment.isin(['baseline','gravity_bias'])].to_string(index=False))


if __name__=='__main__': main()
