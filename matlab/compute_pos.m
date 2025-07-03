% Author: Lauro Ojeda, 2003-2015
function [result] = compute_pos(W,A,PERIOD,USE_KF,W_FF,A_FF,T_FF,MAX_T_FF,FF)
% [result] = compute_pos(W,A,PERIOD,USE_KF,W_FF,A_FF,T_FF,MAX_T_FF,FF)
% 
% Computes IMU positions assuming zero velocity updates
% 
% result: a structure containing results fields
% inputs: 
%   W: Angular Velocity, finite difference form (Angular Velocity (rad/sec) * PERIOD = finite integral of angular velocity over a single sample) 
%   A: Acceleration (m/s/s)
%   PERIOD: Sampling Period (seconds)
% Optional Inputs
%   USE_KF: logical, use (default = 1) or do not use (=0) the Tilt Kalman Filter to correct IMU tilt. 
%   W_FF: Upper Threshold for Angular Velocity (deg/sec) for detecting footfalls 
%   A_FF: Upper Threshold for (Acceleration minus Gravity) (m/s/s) for detecting footfalls 
%   T_FF: Minimum Time that must elapse between two footfalls (default 0.4 sec)
%   MAX_T_FF: Maximum Time that can elapse During Rest Periods before looking for a new footfall (seconds; default 3*T_FF) 
%   FF: Logical array for all the samples in W and A, indicating whether they are Footfalls. 
%       Use this to override internal footfall computations, e.g. if
%       footfall detection is not working well, or for IMUs not on the
%       foot. 


% Set Up Optional Arguments
	if(~exist('USE_KF','var') | isempty(USE_KF))
		USE_KF = 1;
	end;
	if(~exist('W_FF','var') | isempty(W_FF))
		W_FF = [];
	end;
	if(~exist('A_FF','var') | isempty(A_FF))
		A_FF = [];
	end;
	if(~exist('T_FF','var') | isempty(T_FF))
        T_FF = []; % Minimum time allowed between two FFs
	end;
  if(~exist('MAX_T_FF','var') | isempty(MAX_T_FF))
		MAX_T_FF = []; % Maximum rest period allowed before searching for new FFs
  end;
    
% Inertial navigation mechanization
	N = size(W,1);
	t = (1:N)*PERIOD;
	% Compute tilt based on accelerometer readings
	[accel_phi,accel_theta] = acc_tilt(A);
	
	% Determine FFs
  if(~exist('FF','var') | isempty(FF))
		[FF,stationary_periods] = foot_fall(W,A,PERIOD,W_FF,A_FF,T_FF,MAX_T_FF);
	end;

	result.FF = FF;

	% Initialize variables
	quaternion = zeros(N,4);
	An = zeros(N,3);
	Anz = zeros(N,3);
	% Initialize quaternions
	quaternion(1,:) = kf_tilt(PERIOD);
	for(i = 2:N)
		% Compute attitude using quaternion representation
		quaternion(i,:) = qua_est(W(i,:),quaternion(i-1,:));
		% Transform accelerations from body to navigation frame
		rotation_matrix = qua2rot(quaternion(i,:));
		An(i,:) = rotation_matrix*A(i,:)';
		if(USE_KF)
			% Apply KF compensation on tilt
			quaternion(i,:) = kf_tilt(PERIOD,quaternion(i,:),accel_theta(i),accel_phi(i),stationary_periods(i));
		end;
		% Apply Zero Velocity Updates
		% This script has not ben made into a function to make the code run faster
		% this call will update the Az variable wich contains the ZUPT updated navigation acceleration
		zupts;
	end;

	result.Anz = Anz;
	result.An = An;
	result.A = A;
	result.W = W;
	result.quaternion = quaternion;
	
	% Plot attitude
	euler = qua2eul(quaternion);
	result.euler = euler;
	
	figure,
	plot(t,euler*180/pi);
	hold on;grid on;ylabel('Euler angles [deg]');xlabel('time [s]');
	legend('Roll','Pitch','Heading');
	plot(t(FF),euler(FF,:)*180/pi,'*');
	hold off;
	% Set A/V/P to zero during bias drift estimation
	V = cumsum(Anz)*PERIOD;
	Vm = (sum((V(:,1:2).^2)')).^.5;;
	result.V = V;
	result.Vm = Vm;
	% Use speed information to determine the walking section
	try
		result.FF_walking = detect_walking_section(result);
	catch
		result.FF_walking = result.FF; 
	end;
	
	figure,
	plot(t,V,t,Vm,'k');
	hold on;grid on;ylabel('V [m/s]');xlabel('time [s]');
	plot(t(FF),Vm(FF),'*g');
	plot(t(result.FF_walking),Vm(result.FF_walking),'.k');
	legend('Vx','Vy','Vz','|V|','Footfall','Footfall during walk'); 
	hold off;
	
	% Compute and plot positions
	P = cumsum(V)*PERIOD;
	result.P = P;
	
	figure,
	plot3(P(:,1),-P(:,2),-P(:,3));
	axis equal; grid on; hold on;
	plot3(P(FF,1),-P(FF,2),-P(FF,3),'.k');
	xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
	hold off;

