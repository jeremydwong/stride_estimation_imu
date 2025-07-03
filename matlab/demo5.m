% Author: Lauro Ojeda, 2011-2015
%% Two APDM IMUs data processing independently

% Load IMU information from the left foot
[Wb,Ab,PERIOD] = getdata_apdm('20120418-132855_sensor_data_monitor_472_label_Left');
% Plot signals to help determining the SECTION that corresponds to the experiment
getdata(Wb,Ab,PERIOD);

% Define the section (in seconds) that will need to be processed and segment the IMU data accordingly
SECTION = [240,350];
[left_Wb,left_Ab] = getdata(Wb,Ab,PERIOD,SECTION);

% Load IMU information from the right foot
[Wb,Ab,PERIOD] = getdata_apdm('20120418-132857_sensor_data_monitor_403_label_Right.h5');
[right_Wb,right_Ab] = getdata(Wb,Ab,PERIOD,SECTION);

% Perform inertial mechanization for both feet independently
left_walk_info = compute_pos(left_Wb,left_Ab,PERIOD);
right_walk_info = compute_pos(right_Wb,right_Ab,PERIOD);

% Segment strides only for the walking period
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

