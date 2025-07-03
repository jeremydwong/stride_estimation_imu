% Author: Lauro Ojeda, 2011-2015
%% Two APDM IMUs data processing simultaneously



addpath(genpath('D:\Osman\OneDrive - University of Calgary\ResearchRelated\hbcl-all-day-imus-code'));

% Load IMU information from a file
[left_Wb,left_Ab,right_Wb,right_Ab,PERIOD] = sync_apdm('20120418-132855_sensor_data_monitor_472_label_Left.h5','20120418-132857_sensor_data_monitor_403_label_Right.h5');
% Plot signals to help determining the SECTION that corresponds to the experiment
getdata(left_Wb,left_Ab,PERIOD);

% Define the section (in seconds) that will need to be processed and segment the IMU data accordingly
SECTION = [240,350];
[left_Wb,left_Ab] = getdata(left_Wb,left_Ab,PERIOD,SECTION);
[right_Wb,right_Ab] = getdata(right_Wb,right_Ab,PERIOD,SECTION);

% Process data from the two IMUs simultaneously 
[left_walk_info,right_walk_info] = compute_pos_two_imus(left_Wb,left_Ab,right_Wb,right_Ab,PERIOD);

% Segment steps
left_strides = stride_segmentation(left_walk_info,PERIOD);
right_strides = stride_segmentation(right_walk_info,PERIOD);

% Plot results for left and right foot
figure;
subplot(1,2,1);
plt_ltrl_frwd_strides(left_strides,0);
subplot(1,2,2);
plt_ltrl_frwd_strides(right_strides,0);
matchaxes;

figure;
subplot(2,1,1)
plt_frwd_elev_strides(left_strides,0);
subplot(2,1,2)
plt_frwd_elev_strides(right_strides,0);
matchaxes;

figure;
subplot(1,2,1)
plt_stride_var(left_strides,0);
subplot(1,2,2)
plt_stride_var(right_strides,0);
matchaxes;


