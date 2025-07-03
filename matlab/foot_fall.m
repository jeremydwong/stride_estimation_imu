% Author: Lauro Ojeda, 2003-2015
function [FF,stationary_periods] = foot_fall(W,A,PERIOD,W_FF,A_FF,T_FF,MAX_T_FF);
PLOT_DETAILS = 1;
GRAVITY=9.80297286843;
% FF determination settings
if(~exist('W_FF','var') | isempty(W_FF))
	W_FF = 30; % Threshold rate used to determine FFs--ORIGINAL
%     W_FF = 60; % Threshold rate used to determine
end;
if(~exist('A_FF','var') | isempty(A_FF))
	A_FF = 1; %Threshold acceleration used to determine FFs
end;
if(~exist('T_FF','var') | isempty(T_FF))
	T_FF = floor(0.4/PERIOD); % Minimum time allowed between two FFs--ORIGINAL
else
    T_FF = floor(T_FF/PERIOD); % convert Seconds into Samples
end;
if(~exist('MAX_T_FF','var') | isempty(MAX_T_FF))
    MAX_T_FF = T_FF*3; % Maximum rest period allowed before searching for new FFs--ORIGINAL
else
    MAX_T_FF = floor(MAX_T_FF/PERIOD) ; % if input, it is in units of Seconds
end;

Wm = (sum((W.^2)')).^.5*180/pi/PERIOD;
Am = (sum((A.^2)')).^.5 - GRAVITY;

% Find low dynamic conditions
low_motion=find(Wm<W_FF & abs(Am)<A_FF);
% stationary_periods is used by the tilt compensation KF 
N=size(W,1);
stationary_periods = zeros(N,1);
stationary_periods(low_motion) = 1;

% Detect low dynamic segments separated at least T_FF
diff_low_motion = diff(low_motion);
valid_FF = find(diff_low_motion>T_FF);
start_FF = low_motion([1,valid_FF+1]);
end_FF = low_motion([valid_FF,size(low_motion,2)]);

% Find best FF point
FF_size = size(start_FF,2);
FF_index = [];
for(i = 1:FF_size)
	Wm_cut = Wm(start_FF(i):end_FF(i));
	N_cut = size(Wm_cut,2);
	R_seg = floor(N_cut/MAX_T_FF);
	% Segment long low dynamic intervals to find more than one FF
	Wm_seg = reshape(Wm_cut(1:R_seg*MAX_T_FF),MAX_T_FF,R_seg);
	[aux,min_idx] = min(Wm_seg);
	[aux,min_idx2] = min(Wm_cut(MAX_T_FF*R_seg+1:N_cut));
	min_idx = [min_idx,min_idx2];
	FF_cut = min_idx+((0:size(min_idx,2)-1)*MAX_T_FF)+start_FF(i);
	FF_index = [FF_index;FF_cut'];
end;

% Force the last sample to be a FF
FF_index = [FF_index-1;N];
% Create FF boolean vector
FF = zeros(N,1); FF(FF_index) = 1;
% Make FF a logical variable
FF = FF == 1;

t=(1:N)*PERIOD;
% Make plots
clear sh;
figure;

if(PLOT_DETAILS)
subplot(2,1,1);
hold on;
plot(t(low_motion),Wm(low_motion),'.y');
plot(t(start_FF),Wm(start_FF),'og');
plot(t(end_FF),Wm(end_FF),'or');
hold off;
subplot(2,1,2);
hold on;
plot(t(low_motion),Am(low_motion),'.y');
plot(t(start_FF),Am(start_FF),'og');
plot(t(end_FF),Am(end_FF),'or');
hold off;
end;

sh(1) = subplot(2,1,1);
plot(t,Wm);grid on;ylabel('W [deg/sec]');
hold on;
title('Foot fall detection');
plot(t(FF),Wm(FF),'*k');
legend('Signal','Foot-fall');
hold off;

sh(2) = subplot(2,1,2);plot(t,Am);
grid on;ylabel('A [m/sec^2]');
hold on;
plot(t(FF),Am(FF),'*k');
xlabel('t [sec]');
linkaxes(sh,'x');
hold off;
