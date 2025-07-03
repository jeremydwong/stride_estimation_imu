% DIR = '/home/lojeda/Samples/uofm/hbcl/exper/GaitVariabilityAndBalance/ElderSubjects/09072012/';
restoredefaultpath;
close all
DIR = 'C:\Users\odarici\Desktop\spare\SteppingOnFoamImu\v2\v3_ledStrip\MM_ledStrip\';
% addpath('C:\Users\odarici\Desktop\spare\SteppingOnFoamImu\stride_estimation');

L_FILE = 'MM_ledStrip_monitor_780_label_left_20160422-165358.h5';
R_FILE = 'MM_ledStrip_monitor_779_label_right_20160422-165400.h5';

FILE1 = sprintf('%s%s',DIR,L_FILE);
FILE2 = sprintf('%s%s',DIR,R_FILE);

DATA_IS_SYNCED = false

if(DATA_IS_SYNCED)

	[Wb1,Ab1,Wb2,Ab2,PERIOD] = sync_apdm(FILE1,FILE2); LeftWb,LeftAb,RightWb,RightAb,PERIOD,LeftMb,RightMb, static_period


	% Verify that the data is synced by looking at Y-axis rates, you should find that the end of one stride coincides with the beginning of the next one on the other file
	% If the data is not synced, follow the next procedure
else
	% If it does not get synchronized:
	% 1: set  SYNC = 0 in sync_apdm
	% 2: Select a valid window of walking data and use:
	SYNC = 0;
	[W1,A1,W2,A2,PERIOD] = sync_apdm(FILE1,FILE2,SYNC);

	%% IN MOST CASES SKIPPING THIS AND USING THE WHOLE FILE GIVES THE BEST RESULTS
	if(0)
        temp_xlim = xlim;
		SECTION = [1000 1600];
		[W1,A1]= getdata(W1,A1,PERIOD,SECTION);
		[W2,A2]= getdata(W2,A2,PERIOD,SECTION);
	end;
	%% END OF SKIP 

	% In some cases, with SECTION  = [], I get better results
% 	FORCE_SYNC_VALUE = find_data_shift(W1,W2,PERIOD);
% 	4: set  SYNC = 1 
	% 5: Set the FORCE_SYNC_VALUE to the result of this function
	% Run this tool again;
	SYNC = 1;
% 	[Wb1,Ab1,Wb2,Ab2,PERIOD] = sync_apdm(FILE1,FILE2,SYNC,FORCE_SYNC_VALUE);

%     W = Wb1; A = Ab1; FILE = [FILE1(1:end-2),'mat']; save(FILE,'W','A','PERIOD');
%     W = Wb2; A = Ab2; FILE = [FILE2(1:end-2),'mat']; save(FILE,'W','A','PERIOD');
    
	% In the worst case, the FORCE_SYNC_VALUE can be defended manually as follows
	FORCE_SYNC_VALUE = 614.7%390%; % Override 
	[Wb1,Ab1,Wb2,Ab2,PERIOD] = sync_apdm(FILE1,FILE2,SYNC,FORCE_SYNC_VALUE);
    5
    
        W = Wb1; A = Ab1; FILE = [FILE1(1:end-2),'mat']; save(FILE,'W','A','PERIOD');
    W = Wb2; A = Ab2; FILE = [FILE2(1:end-2),'mat']; save(FILE,'W','A','PERIOD');
end
 

