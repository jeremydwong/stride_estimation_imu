% Author: Lauro Ojeda, 2012-2015
function [LeftWb,LeftAb,RightWb,RightAb,PERIOD,LeftMb,RightMb, static_period] = sync_apdm(L_FILE,R_FILE,SYNC,FORCE_SYNC_VALUE); %<--------------static_period
if(~exist('SYNC','var'))
	SYNC = 1; 
end;
if(~exist('FORCE_SYNC_VALUE','var'))
	FORCE_SYNC_VALUE = 0;% Use a non-zero value (counts) combined with SYNC=1 in order to force data synchronization
end;
caseIdList = hdf5read(L_FILE,'/CaseIdList');
groupName = caseIdList(1).data;
L_time = hdf5read(L_FILE, [groupName '/Time']);
FREQ = hdf5read(L_FILE, [groupName '/SampleRate']);
PERIOD = 1/double(FREQ);

caseIdList = hdf5read(R_FILE,'/CaseIdList');
groupName = caseIdList(1).data;
R_time = hdf5read(R_FILE, [groupName '/Time']);
L_SECTION = [1,size(L_time,1)]*PERIOD;
R_SECTION = [1,size(R_time,1)]*PERIOD;
if(SYNC)
	if(FORCE_SYNC_VALUE)
		% If a known value of the time shift is know, replace the variable FORCE_SYNC_VALUE 
		% with the correct value, you may use croscorrelations to find the correct shift value
		warning('Sync with user defined value');
		shift = FORCE_SYNC_VALUE;
	else
		display('Sync with sensor timer');
		if(L_time(1) <= R_time(1))
			shift = -find(L_time >= R_time(1),1)
		else
			shift = find(R_time >= L_time(1),1)
		end;
	end;
	if(shift<0)
		display(sprintf('Shift Left IMU signals [%d]',-shift));
		L_SECTION = [-shift,size(L_time,1)]*PERIOD; % To shift left side
	else
		display(sprintf('Shift Right IMU signals [%d]',shift));
		R_SECTION = [shift,size(R_time,1)]*PERIOD; % To shift right side
	end;
else
		warning('Not Syncing');
end;

% Load data 
[Wb,Ab,PERIOD,Mb] = getdata_apdm(L_FILE,1);
[LeftWb,LeftAb, static_period, LeftMb] = getdata(Wb,Ab,PERIOD,L_SECTION,[],Mb); %<--------------static_period
set(gcf,'Name','Left IMU');
[Wb,Ab,PERIOD,Mb] = getdata_apdm(R_FILE,1);
[RightWb,RightAb, ~, RightMb] = getdata(Wb,Ab,PERIOD,R_SECTION,[],Mb); %<--------------static_period
set(gcf,'Name','Right IMU');
