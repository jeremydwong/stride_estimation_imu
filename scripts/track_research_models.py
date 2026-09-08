"""Research controls for debug_track_extended.py; not production estimators.

The ESKF uses global attitude errors as described in Solà (2017), section 7:
https://arxiv.org/html/1711.02508v1#S7
The state is position, velocity, navigation-frame attitude error, gyro bias,
and accelerometer bias. Gravity is fixed; zero-velocity observations occur
at contacts or across the low-motion mask. The optional floor measurement is
an explicit z=0 prior. These controls did NOT outperform the stride estimator;
noise settings are exploratory per-sample standard deviations, not sensor
specification estimates. Returned positions include Kalman feedback jumps.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R, Rotation
from scipy.linalg import solve
from stride_imu.inertial import foot_fall, GRAVITY
from stride_imu.experimental import kalman_filter_gravity
from types import SimpleNamespace

def skew(a):
 x,y,z=a;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def eskf(W,A,dt,bias,init_a,*,dense=False,floor=False,bias_sigma=.1,acc_sigma=.05,zupt_sigma=.03,gyro_noise=.01,acc_noise=.5,trap=True):
 n=len(W);FF,stationary=foot_fall(W,A,dt);p=np.zeros((n,3));v=np.zeros_like(p);q0,_=kalman_filter_gravity([1.,0,0,0],init_a,1e12)
 rot=R.from_quat(q0[[1,2,3,0]]);bg=bias.copy();ba=np.zeros(3)
 P=np.diag(np.r_[np.full(3,.01**2),np.full(3,.01**2),np.full(3,np.deg2rad(3)**2),np.full(3,np.deg2rad(bias_sigma)**2),np.full(3,acc_sigma**2)])
 Q=np.diag(np.r_[np.zeros(3),np.full(3,acc_noise**2*dt**2),np.full(3,gyro_noise**2*dt**2),np.full(3,1e-12*dt),np.full(3,1e-8*dt)])
 H=np.zeros((4 if floor else 3,15));H[:3,3:6]=np.eye(3)
 if floor:H[3,2]=1
 noise=np.diag([zupt_sigma**2]*3+([.02**2] if floor else []));Is=np.eye(15)
 bg_hist=np.zeros((n,3));ba_hist=np.zeros((n,3));rot_hist=np.zeros((n,4));rot_hist[0]=rot.as_quat()
 for i in range(1,n):
  w=(W[i]+W[i-1])/2 if trap else W[i]
  rot=rot*R.from_rotvec(w-bg*dt);C=rot.as_matrix();an=C@(A[i]-ba);net=an+[0,0,GRAVITY]
  p[i]=p[i-1]+v[i-1]*dt+.5*net*dt**2;v[i]=v[i-1]+net*dt
  F=Is.copy();F[:3,3:6]=np.eye(3)*dt;F[3:6,6:9]=-skew(an)*dt;F[3:6,12:15]=-C*dt;F[6:9,9:12]=-C*dt
  F[:3,6:9]=-.5*skew(an)*dt**2;F[:3,12:15]=-.5*C*dt**2
  P=F@P@F.T+Q
  if (stationary[i] if dense else FF[i]):
   residual=np.r_[-v[i],-p[i,2]] if floor else -v[i]
   K=solve(H@P@H.T+noise,H@P,assume_a='pos').T;dx=K@residual
   p[i]+=dx[:3];v[i]+=dx[3:6];rot=R.from_rotvec(dx[6:9])*rot;bg+=dx[9:12];ba+=dx[12:15]
   IKH=Is-K@H;P=IKH@P@IKH.T+K@noise@K.T
   reset=Is.copy();reset[6:9,6:9]=np.eye(3)+.5*skew(dx[6:9]);P=reset@P@reset.T
  bg_hist[i]=bg;ba_hist[i]=ba;rot_hist[i]=rot.as_quat()
 return SimpleNamespace(P=p,V=v,FF=FF,bg=bg_hist,ba=ba_hist,q=rot_hist)


def stadium(s,radius=12.):
 straight=(200-2*np.pi*radius)/2;s=np.asarray(s);p=np.zeros((len(s),2));heading=np.zeros(len(s))
 m=s<straight;p[m,1]=s[m];heading[m]=np.pi/2
 m=(s>=straight)&(s<straight+np.pi*radius);a=(s[m]-straight)/radius;p[m]=np.c_[radius-radius*np.cos(a),straight+radius*np.sin(a)];heading[m]=np.pi/2-a
 m=(s>=straight+np.pi*radius)&(s<2*straight+np.pi*radius);p[m]=np.c_[np.full(sum(m),2*radius),2*straight+np.pi*radius-s[m]];heading[m]=-np.pi/2
 m=s>=2*straight+np.pi*radius;a=(s[m]-2*straight-np.pi*radius)/radius;p[m]=np.c_[radius+radius*np.cos(a),-radius*np.sin(a)];heading[m]=-np.pi/2-a
 return p,heading

def synthetic(side,dt=1/128, radius=12., rise=0.):
 t=np.arange(0,133,dt);phase=t-2-(.5 if side=='right' else 0);phase=np.clip(phase,0,128);step=np.floor(phase);u=np.clip((phase-step-.5)/.5,0,1);blend=6*u**5-15*u**4+10*u**3
 distance=(step+blend)*200/128;p2,yaw=stadium(distance,radius);p=np.c_[p2,.12*np.sin(np.pi*u)**4+rise*distance/200];mount=-.4;swing=1.5 if side=='right' else 1.1
 angles=np.c_[np.full(len(t),-.15 if side=='right' else .15),mount-swing*np.sin(np.pi*u)**2,yaw+(.06 if side=='right' else -.06)]
 rot=Rotation.from_euler('xyz',angles);v=np.gradient(p,dt,axis=0);acc=np.gradient(v,dt,axis=0);A=rot.inv().apply(acc+[0,0,-GRAVITY]);W=np.zeros_like(A);W[1:]=(rot[:-1].inv()*rot[1:]).as_rotvec()
 bias=np.deg2rad([-.4,.03,.05] if side=='right' else [.12,0,-.12]);W+=bias*dt
 return t,W,A,p,bias,A[0]
