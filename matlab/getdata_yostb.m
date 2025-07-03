% Author: Lauro Ojeda, 2013-2015
function [W,A,PERIOD,M] = getdata_yostb(FILE)
PLOT_DETAILS = 0;
PACKET_SIZE = 42; %1B(Hours)%1B(Minutes)%4F(Seconds)%4F(NormAccelX)%4F(NormAccelY)%4F(NormAccelZ)%4F(GyroX)%4F(GyroY)%4F(GyroZ)%4F(RawCompassX)%4F(RawCompassY)%4F(RawCompassZ)"
GRAVITY = 9.80297286843;
if(~exist('FILE','var')) FILE = 'data.dat'; end;
fid = fopen(FILE);
aux = fgets(fid); %disregard header
[data,N] = fread(fid,Inf,'uint8=>uint8');
fclose(fid);
MAX_VALID = floor((N)/PACKET_SIZE);
packets = reshape(data(1:MAX_VALID*PACKET_SIZE),PACKET_SIZE,MAX_VALID);
t = zeros(MAX_VALID,1);
wx = zeros(MAX_VALID,1);
wy = zeros(MAX_VALID,1);
wz = zeros(MAX_VALID,1);
ax = zeros(MAX_VALID,1);
ay = zeros(MAX_VALID,1);
az = zeros(MAX_VALID,1);
mx = zeros(MAX_VALID,1);
my = zeros(MAX_VALID,1);
mz = zeros(MAX_VALID,1);
for(i = 1:MAX_VALID)
	t(i) = typecast(packets(6:-1:3,i),'single');
	ax(i) = typecast(packets(10:-1:7,i),'single');
	ay(i) = typecast(packets(14:-1:11,i),'single');
	az(i) = typecast(packets(18:-1:15,i),'single');
	wx(i) = typecast(packets(22:-1:19,i),'single');
	wy(i) = typecast(packets(26:-1:23,i),'single');
	wz(i) = typecast(packets(30:-1:27,i),'single');
	mx(i) = typecast(packets(34:-1:31,i),'single');
	my(i) = typecast(packets(38:-1:35,i),'single');
	mz(i) = typecast(packets(42:-1:39,i),'single');
end
% Swap axis
% X forward towards the LED, Y Right, Z Down
WX = -wz; WY = -wx; WZ = wy;
AX =  az; AY =  ax; AZ =-ay;
MX =  mz; MY =  mx; MZ = my;

dt=diff(t);
neg=find(dt<-50);
dt(neg)=dt(neg)+60;

% Compute average sampling PERIOD
PERIOD=median(dt);

W = [WX,WY,WZ]*PERIOD;
A = [AX,AY,AZ]*GRAVITY;
M = [MX,MY,MZ];

% Find samples that were captured after skiping many samples 
if(PLOT_DETAILS)
	OUTLIER_SCALE = 5;
	outlier_threshold=std(dt)*OUTLIER_SCALE;
	outlier=find(abs(dt-PERIOD)>outlier_threshold);
	disp(['Warning outliers = ',int2str(length(outlier))]);
	time=cumsum(dt)/60;
	figure,
	plot(time,dt);
	hold on;
	plot(time(outlier),dt(outlier),'.r');
	xlabel('Time [min]');
	ylabel('Period [sec]');
	disp([1/PERIOD, 1/mean(dt)]);
end;

