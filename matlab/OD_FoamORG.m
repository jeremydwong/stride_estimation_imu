%% Syncronize

clear, close all
[LeftWb,LeftAb,RightWb,RightAb,PERIOD] = sync_apdm('OD_30Oct14_monitor_472_label_Left_20141030-175712.h5','OD_30Oct14_monitor_403_label_Right_20141030-175709.h5');

save ODFoam_AllData
%%
% sections
clear
close all

load ODFoam_AllData
FILTER = 0;
% 
SECTION = [1568 1740]; % Everything
OutlierSec = []; % 

[left_Wb,left_Ab] = getdata(LeftWb,LeftAb,PERIOD,SECTION);
[right_Wb,right_Ab] = getdata(RightWb,RightAb,PERIOD,SECTION);

% Process data from the two IMUs on the feet simultaneously 
[Left_foot,Right_foot] = compute_pos_two_imus(left_Wb,left_Ab,right_Wb,right_Ab,PERIOD);

% Segment steps

left_strides = stride_segmentation(Left_foot,PERIOD,FILTER,OutlierSec);
right_strides = stride_segmentation(Right_foot,PERIOD,FILTER,OutlierSec);

%%
clear
% close all

load ODFoam_AllData
FILTER = 0;

SECTION = [1568 1740]; % Level Walking 

OutlierSec_L = [.1 40.13
                 80.26 86.78
                 147.06 153.91
                 167.66 171.80]; %
OutlierSec_R = [.1 40.13
                 80.26 86.78
                 147.06 153.91
                 167.66 171.80]; % 
           
[left_Wb,left_Ab] = getdata(LeftWb,LeftAb,PERIOD,SECTION);
[right_Wb,right_Ab] = getdata(RightWb,RightAb,PERIOD,SECTION);

% Process data from the two IMUs on the feet simultaneously 
[Left_foot,Right_foot] = compute_pos_two_imus(left_Wb,left_Ab,right_Wb,right_Ab,PERIOD);

% Segment steps

left_strides = stride_segmentation(Left_foot,PERIOD,FILTER,OutlierSec_L);
right_strides = stride_segmentation(Right_foot,PERIOD,FILTER,OutlierSec_R);

save ODFoam_Level

%%
clear
close all

load ODFoam_AllData
FILTER = 0;

SECTION = [1998 2040]; % Stepping on Foam

OutlierSec_L = [0.1 20.09
                31.79 42.00]; %
OutlierSec_R = [0.1 20.09
                31.79 42.00]; % %
[left_Wb,left_Ab] = getdata(LeftWb,LeftAb,PERIOD,SECTION);
[right_Wb,right_Ab] = getdata(RightWb,RightAb,PERIOD,SECTION);

% Process data from the two IMUs on the feet simultaneously 
[Left_foot,Right_foot] = compute_pos_two_imus(left_Wb,left_Ab,right_Wb,right_Ab,PERIOD);

% Segment steps

left_strides = stride_segmentation(Left_foot,PERIOD,FILTER,OutlierSec_L);
right_strides = stride_segmentation(Right_foot,PERIOD,FILTER,OutlierSec_R);

time = (1:length(Left_foot.P(:,3)))';
time = time*PERIOD;


a = round(20.09/PERIOD);
b = round(31.79/PERIOD);

figure
plot(time(a:b,1),-Left_foot.P(a:b,2))
% axis([0 42 -1 1])
 
save ODFoam_SteppingonFoam

%% Plot the results for left and right foot
figure('Color',[1,1,1]);
subplot(1,2,1);
plt_ltrl_frwd_strides(left_strides);
hold on
% plot(mean(left_strides.ltrl'),mean(left_strides.frwd'),'k','LineWidth',3)
% axis equal, box on
axis([-.4 .4 0 2])

subplot(1,2,2);
plt_ltrl_frwd_strides(right_strides);
hold on
% plot(mean(right_strides.ltrl'),mean(right_strides.frwd'),'k','LineWidth',3)
% axis equal, box on
axis([-.4 .4 0 2])

figure('Color',[1,1,1]);
subplot(2,1,1)
plt_frwd_elev_strides(left_strides,0);
hold on
% plot(mean(left_strides.frwd'),-mean(left_strides.elev'),'k','LineWidth',3)
% axis equal, box on
axis([0 2 -0.6 0.6])

subplot(2,1,2)
plt_frwd_elev_strides(right_strides,0);
hold on
% plot(mean(right_strides.frwd'),-mean(right_strides.elev'),'k','LineWidth',3)
% axis equal, box on
% axis([0 1.5 0 .5])
axis([0 2 -0.6 .6])

figure('Color',[1,1,1]);
subplot(1,2,1)
plt_stride_var(left_strides,0);
% axis equal, box on
% axis([-0.25 0.25 -0.25 0.25])

subplot(1,2,2)
plt_stride_var(right_strides,0);
% axis equal, box on
% axis([-0.25 0.25 -0.25 0.25])

