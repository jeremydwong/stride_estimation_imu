% Author: Lauro Ojeda, 2011-2015
function [left_walk_info,right_walk_info] = compute_pos_two_imus (leftWb,leftAb,rightWb,rightAb,PERIOD,USE_KF,W_FF,A_FF,FFL,FFR)
	if(~exist('USE_KF','var') | isempty(USE_KF))
		USE_KF = 1;
	end;
	if(~exist('W_FF','var') | isempty(W_FF))
		W_FF = [];
	end;
	if(~exist('A_FF','var') | isempty(A_FF))
		A_FF = [];
	end;
	if(~exist('FFL','var') | isempty(FFL))
		FFL = [];
	end;
	if(~exist('FFR','var') | isempty(FFR))
		FFR = [];
	end;
	PLOT_DETAILS = 0;
	%% Compute individual foot paths
	% This is the file containing both feet IMU data
	% Process each foot individually
	% left_walk_info foot
	left_walk_info = compute_pos(leftWb,leftAb,PERIOD,USE_KF,W_FF,A_FF,FFL);

	% right_walk_info foot
	right_walk_info = compute_pos(rightWb,rightAb,PERIOD,USE_KF,W_FF,A_FF,FFR);
	% Verify that both IMU's collected the same amount of information
	left_samples = size(left_walk_info.FF,1);
	right_samples = size(right_walk_info.FF,1);
	if(left_samples ~= right_samples)
		error(sprintf('Files may not be synced L = % d R = % d, define a SECTION',left_samples,right_samples))
	end;
	%% Merge/Combine previous FF detection with new one based on cross-velocity
	% Notice that the footfall depneds on information of the opposite foot
	[left_walk_info.FF,left_walk_info.FF_walking,left_walk_info.FF_max_speed] = foot_fall_opposite_velocity(right_walk_info.Vm,left_walk_info.FF,PERIOD);
	[right_walk_info.FF,right_walk_info.FF_walking,right_walk_info.FF_max_speed] = foot_fall_opposite_velocity(left_walk_info.Vm,right_walk_info.FF,PERIOD);

	SAMPLES = left_samples;
	t = (1:SAMPLES)*PERIOD;

	MAX_NUMBER_STEP_DIFF = 2;%Default 2
	if(abs(length(find(left_walk_info.FF_walking)) - length(find(right_walk_info.FF_walking))) > MAX_NUMBER_STEP_DIFF)
		disp('Warning! incompatible number of steps, this needs to be fixed.\nCheck MIN_WALK_SPEED in foot_fall_opp..');
		disp([length(find(left_walk_info.FF_walking)) length(find(right_walk_info.FF_walking))]);
		pause;
	end;

	%% Recompute accelerations (ZUPT) using combined FFs
	% left_walk_info foot
	An = left_walk_info.An;
	FF = left_walk_info.FF;
	Anz = zeros(SAMPLES,3);
	for(i = 2:SAMPLES) zupts; end;
	left_walk_info.Az = Anz;
	left_walk_info.V = cumsum(Anz)*PERIOD;
	left_walk_info.P = cumsum(left_walk_info.V)*PERIOD;
	left_walk_info.Vm = (sum((left_walk_info.V.^2)')).^.5;
	% right_walk_info foot
	An = right_walk_info.An;
	FF = right_walk_info.FF;
	Anz = zeros(SAMPLES,3);
	for(i = 2:SAMPLES) zupts; end;
	right_walk_info.Az = Anz;
	right_walk_info.V = cumsum(Anz)*PERIOD;
	right_walk_info.P = cumsum(right_walk_info.V)*PERIOD;
	right_walk_info.Vm = (sum((right_walk_info.V.^2)')).^.5;

	%% Make Plots
	figure;
	hold on;
	plot3(left_walk_info.P(left_walk_info.FF,1),left_walk_info.P(left_walk_info.FF,2),left_walk_info.P(left_walk_info.FF,3),'.g');
	plot3(right_walk_info.P(right_walk_info.FF,1),right_walk_info.P(right_walk_info.FF,2),right_walk_info.P(right_walk_info.FF,3),'.r');
	legend('left_walk_info','right_walk_info');
	plot3(left_walk_info.P(:,1),left_walk_info.P(:,2),left_walk_info.P(:,3));
	plot3(right_walk_info.P(:,1),right_walk_info.P(:,2),right_walk_info.P(:,3));
	xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
	axis equal; grid on;
	hold off;

	figure;
	sh(1) = subplot(2,1,1);plot(t,left_walk_info.Vm);
	hold on;grid on;ylabel('Left |V| [m/s]');xlabel('time [s]');
	plot(t(left_walk_info.FF),left_walk_info.Vm(left_walk_info.FF),'g*');
	plot(t(left_walk_info.FF_walking),left_walk_info.Vm(left_walk_info.FF_walking),'k.');
	hold off;
	legend('|V|','FFs','Walking');
	title('Foot Fall detection using opposite shoe speed');
	sh(2) = subplot(2,1,2);plot(t,right_walk_info.Vm);
	hold on;grid on;ylabel('Right |V| [m/s]');xlabel('time [s]');
	plot(t(right_walk_info.FF),right_walk_info.V(right_walk_info.FF),'g*');
	plot(t(right_walk_info.FF_walking),right_walk_info.V(right_walk_info.FF_walking),'k.');
	hold off;
	linkaxes(sh,'x');

	if(PLOT_DETAILS)
		left_az_mag = (sum((left_walk_info.Az.^2)')).^.5;
		right_az_mag = (sum((right_walk_info.Az.^2)')).^.5;
		figure;
		sh(1) = subplot(2,1,1);plot(t,left_walk_info.Az,t,left_az_mag,'k');hold on;grid on;ylabel('left_walk_info A [m/s^2]');xlabel('time [s]');
		plot(t(left_walk_info.FF),left_walk_info.V(left_walk_info.FF),'g.');
		plot(t(left_walk_info.FF_walking),left_walk_info.V(left_walk_info.FF_walking),'yo');
		sh(2) = subplot(2,1,2);plot(t,right_walk_info.Az,t,right_az_mag,'k');hold on;grid on;ylabel('right_walk_info A [m/s^2]');xlabel('time [s]');
		plot(t(right_walk_info.FF),right_walk_info.V(right_walk_info.FF),'g.');
		plot(t(right_walk_info.FF_walking),right_walk_info.V(right_walk_info.FF_walking),'yo');
		hold off;
		linkaxes(sh,'x');
	end;


