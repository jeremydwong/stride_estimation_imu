% Author: Lauro Ojeda, 2011-2015
%% Two MemeSense IMUs data processing simultaneously
% Load IMU information from the left foot
[Wb,Ab,PERIOD] = getdata_ms('LOG012U1.dat');

% Plot signals to help determining the SECTION that corresponds to the experiment
getdata(Wb,Ab,PERIOD);

% Define the section (in seconds) that will need to be processed and segment the IMU data accordingly
SECTION = [170,233];
[left_Wb,left_Ab] = getdata(Wb,Ab,PERIOD,SECTION);

% Load IMU information from the right foot
[Wb,Ab] = getdata_ms('LOG012U0.dat');

% The section for the right foot should be the same as the left foot
[right_Wb,right_Ab] = getdata(Wb,Ab,PERIOD,SECTION);

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

