% Author: Lauro Ojeda, 2001-2015
function [quaternion] = kf_tilt(PERIOD,quaternion,accel_theta,accel_phi,is_stationary)
	% Tilt error compensation using accelerometer - based tilt
	% A one state indirect KF (complementary filter) is used for the process and the measurements are applied only on low
	% motion conditions (is_stationary)
	% KF constants
	persistent P Q R
	if(~exist('quaternion','var'))
		% Initialize KF
		P = 0; % Initial covariance error
		Q = 1.5e-5*PERIOD; % Process (gyros) covariance error
		R = 1.5e-1*PERIOD; % Measurement (accelerometer) covariance error
		quaternion = [1,0,0,0];
		return;
	end;
	% Propagate covariance error
	P = P + Q;
	if(is_stationary)
		% Compute Kalman gain
		K = P/(P + R);
		euler = qua2eul(quaternion);	
		% Apply corrections to the state
		quaternion = eul2qua([euler(1) - (euler(1) - accel_phi)*K,euler(2) - (euler(2) - accel_theta)*K,euler(3)]);
		% Update covariance error
		P = (1 - K)*P*(1 - K) + K*R*K;
	end;

