% Author: Lauro Ojeda, 2008-2015
function [W,A,PERIOD]  =  getdata_ms(FILE,ORIENTATION)
	%Loads parameters and transforms the data to the correct data axes
	FREQ = 150;
	GRAVITY = 9.80297286843;
	if(~exist('FILE','var')) FILE = 'data.dat'; end;
	data = readLogomaticMemsenseFile(FILE);
	w = data.gyroData*pi/180;
	a = data.accelData*GRAVITY;

	XFWD_YRGT_ZDWN = 1; %Original axes
	XDWN_YLFT_ZFRW = 2; %On shoe counter (back) 
	XBCK_YLFT_ZDWN = 3; %Inside shoe heel
	XFWD_YUPW_ZRGT = 4; %With mount, IMU outside shoe (RIGHT)
	XFWD_YDWN_ZLFT = 5; %With mount, IMU outside shoe (LEFT)
	XBCK_YLFT_ZUPW = 6; %Original with X-axis pointing backwards

	if(~exist('ORIENTATION','var')) ORIENTATION = XFWD_YRGT_ZDWN; end;
	switch(ORIENTATION)
	case XFWD_YRGT_ZDWN 
		WX =  w(:,1); WY =  w(:,2); WZ =  w(:,3);
		AX = -a(:,1); AY = -a(:,2); AZ = -a(:,3);
	case XDWN_YLFT_ZFRW
		WX =  w(:,3); WY = -w(:,2); WZ =  w(:,1);
		AX = -a(:,3); AY = -a(:,2); AZ = -a(:,1); 
	case XBCK_YLFT_ZDWN  
		WX = -w(:,1); WY = -w(:,2); WZ =  w(:,3);
		AX =  a(:,1); AY =  a(:,2); AZ = -a(:,3); 
	case XFWD_YUPW_ZRGT
		WX =  w(:,1); WY = -w(:,3); WZ =  w(:,2);
		AX = -a(:,1); AY =  a(:,3); AZ = -a(:,2); 
	case XFWD_YDWN_ZLFT
		WX =  w(:,1); WY =  w(:,3); WZ = -w(:,2);
		AX = -a(:,1); AY = -a(:,3); AZ =  a(:,2);
	case XBCK_YLFT_ZUPW
		WX = -w(:,1); WY = -w(:,2); WZ =  w(:,3);
		AX =  a(:,1); AY =  a(:,2); AZ = -a(:,3);
	end;

	PERIOD = 1/FREQ;
	W = [WX,WY,WZ]*PERIOD;
	A = [AX,AY,AZ];

