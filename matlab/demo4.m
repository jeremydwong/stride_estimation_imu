% Author: Lauro Ojeda, 2011-2015
%% Single APDM IMU data processing
% Load IMU information from a file
[Wb,Ab,PERIOD] = getdata_apdm('20120418-132857_sensor_data_monitor_403_label_Right.h5');

% Plot signals to help determining the SECTION that corresponds to the experiment
getdata(Wb,Ab,PERIOD);

% Define the section (in seconds) that will need to be processed and segment the IMU data accordingly
SECTION = [240,350];
[Wb,Ab] = getdata(Wb,Ab,PERIOD,SECTION);

% Perform inertial mechanization
walk_info = compute_pos(Wb,Ab,PERIOD);

% Segment strides only for the walking period
strides = stride_segmentation(walk_info,PERIOD);

% Plot results
plt_ltrl_frwd_strides(strides);
plt_frwd_elev_strides(strides);
plt_stride_var(strides);

