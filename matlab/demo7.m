%% Single YEI IMU data processing
% Load IMU information from a file
[Wb,Ab,PERIOD] = getdata_yei('data_yei.dat');
IMU = 120;
cal_yei;
Wb(:,1) = Wb(:,1)*SFW(IMU-BASE,1);
Wb(:,2) = Wb(:,2)*SFW(IMU-BASE,2);
Wb(:,3) = Wb(:,3)*SFW(IMU-BASE,3);
Ab(:,1) = Ab(:,1)*SFA(IMU-BASE,1);
Ab(:,2) = Ab(:,2)*SFA(IMU-BASE,2);
Ab(:,3) = Ab(:,3)*SFA(IMU-BASE,3);

% Plot signals to help determining the SECTION that corresponds to the experiment
getdata(Wb,Ab,PERIOD);

% Define the section (in seconds) that will need to be processed and segment the IMU data accordingly
SECTION = [50,100];
%SECTION = [121,158];
[Wb,Ab] = getdata(Wb,Ab,PERIOD,SECTION);

% Perform inertial mechanization
walk_info = compute_pos(Wb,Ab,PERIOD);
%walk_info = compute_pos(Wb,Ab,PERIOD, 1, 60, 2);

% Segment strides only for the walking period
strides= stride_segmentation(walk_info,PERIOD);

% Plot results
plt_ltrl_frwd_strides(strides);
plt_frwd_elev_strides(strides);
plt_stride_var(strides);

