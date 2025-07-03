% Author: Lauro Ojeda, 2012-2015
function [W,A,PERIOD,M] = getdata_apdm(FILE,ORIENTATION)
	if(~exist('FILE','var')) FILE = 'data.h5'; end;
	caseIdList = hdf5read(FILE,'/CaseIdList');
	groupName = caseIdList(1).data;
	FREQ = hdf5read(FILE, [groupName '/SampleRate']);
	PERIOD = 1/double(FREQ);

	a = hdf5read(FILE, [groupName '/Calibrated/Accelerometers'])'; %Transposed to make Nx3 in MATLAB
	w = hdf5read(FILE, [groupName '/Calibrated/Gyroscopes'])'; %Transposed to make Nx3 in MATLAB
	m = hdf5read(FILE, [groupName '/Calibrated/Magnetometers'])'; %Transposed to make Nx3 in MATLAB
	time = hdf5read(FILE, [groupName '/Time']);
	ORIGINAL = 0;
	LED_UP_RIGHT_FRWD = 1; % This one is normally used on feet
	LED_UP_LEFT_FRWD = 2;
	LED_LOW_RIGHT_FRWD = 3;
	LED_LOW_DOWN_FRWD = 4;
	LED_UP_RIGHT_BACK = 5; 
	LED_LOW_LEFT_BACK = 6;
	if(~exist('ORIENTATION','var')) ORIENTATION = LED_UP_RIGHT_FRWD; end;
	switch(ORIENTATION)
		case ORIGINAL 
			% +X  <---o +Z
			%         |
			%         V +Y
			WX = w(:,1); WY = w(:,2); WZ = w(:,3);
			AX = a(:,1); AY = a(:,2); AZ = a(:,3);
		case LED_UP_RIGHT_FRWD
			WX = w(:,2); WY = w(:,1); WZ = -w(:,3);
			AX = a(:,2); AY = a(:,1); AZ = -a(:,3);
		case LED_UP_LEFT_FRWD
			WX = w(:,1); WY = -w(:,2); WZ = -w(:,3);
			AX = a(:,1); AY = -a(:,2); AZ = -a(:,3);
		case LED_LOW_DOWN_FRWD
			WX = -w(:,2); WY = -w(:,1); WZ = -w(:,3);
			AX = -a(:,2); AY = -a(:,1); AZ = -a(:,3);
		case LED_LOW_RIGHT_FRWD
			WX = -w(:,1); WY = w(:,2); WZ = -w(:,3);
			AX = -a(:,1); AY = a(:,2); AZ = -a(:,3);
		case LED_UP_RIGHT_BACK
			WX = w(:,3); WY = -w(:,1); WZ = w(:,2);
			AX = a(:,3); AY = -a(:,1); AZ = a(:,2);
		case LED_LOW_LEFT_BACK
			WX = w(:,3); WY = w(:,1); WZ = -w(:,2);
			AX = a(:,3); AY = a(:,1); AZ = -a(:,2);
	end;

	W = [WX,WY,WZ]*PERIOD;
	A = [AX,AY,AZ];
	M = m;
