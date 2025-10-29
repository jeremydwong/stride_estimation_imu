% Author: Lauro Ojeda, 2008-2015
function [W,A, static_period, M]  =  getdata(Win,A,PERIOD,SECTION_SECONDS,BIAS,M) %<------- added static period
% BIAS, is the static time at the beggining of the experiment
% leave BIAS the parameter empty to make the system detect the static time automatically
% 
W = Win;
GRAVITY = 9.80297286843;
if(exist('SECTION_SECONDS','var') & ~isempty(SECTION_SECONDS))
	number_of_sections = size(SECTION_SECONDS,1);
	SECTION_SAMPLES = [];
	for(ii=1:number_of_sections)
		SECTION_SAMPLES  =  [SECTION_SAMPLES,(floor(SECTION_SECONDS(ii,1)/PERIOD):floor(SECTION_SECONDS(ii,2)/PERIOD))];
	end;
	W = W(SECTION_SAMPLES,:);
	A = A(SECTION_SAMPLES,:);
	if(exist('M','var'))
        M = M(SECTION_SAMPLES,:);
    end;
end;
if(exist('BIAS','var') & ~isempty(BIAS)) 
	static_period = (1:BIAS/PERIOD);
else
	static_period = detect_quite_time(W,PERIOD);
end;

% Apply gyro static bias compensation
W = W-ones(size(W,1),1)*mean(W(static_period,:));

% Normalizes accelerometer during static time using a 1-G compensation
static_acceleration = sum((A(static_period,:).^2)').^.5;
gravity_measurement = mean(static_acceleration);
A = A*GRAVITY/gravity_measurement;

if(exist('M','var') & ~isempty(M)) 
	include_magnetomter = 1;
else
	include_magnetomter = 0;
end;

t = (1:size(W,1))*PERIOD;

figure,
sh_xa(1) = subplot(2 + include_magnetomter,1,1);
hold on;
plot(t(static_period),W(static_period,:)/PERIOD*180/pi,'.k');
legend('Static time');
plot(t,W/PERIOD*180/pi);
grid on;
ylabel('W [deg/s]');
title('Body referenced inertial signals');
hold off;

sh_xa(2) = subplot(2 + include_magnetomter,1,2);plot(t,A);
grid on;ylabel('A [m/s^2]');

if(include_magnetomter)
	sh_xa(3) = subplot(3,1,3);plot(t,M); grid on;ylabel('Mag');
end;

linkaxes(sh_xa,'x');
xlabel('time [s]');
