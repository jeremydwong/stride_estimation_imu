% Author: Lauro Ojeda, 2011-2015
function [FF,FF_walking,FF_max_speed] = foot_fall_opposite_velocity(Vm,FF_org,PERIOD)
	MERGE_MODE = 'MAX_SPEED_OR_ORG';
	PLOT_DETAILS = 0;
	WALK_SPEED_PERCENTAGE  =  0.8; % Determines the first and last foot fall, use a small value to include more flutuations in speed. For normal walk use .9, for varying speed use .5
	MIN_FF_SEPARATION = floor((1/PERIOD)/2.5);
	MIN_WALK_SPEED= 2; % minimun foot velocity while walking, Use 2 for normal walk, 1.2 for varing speeds
	ACCEL_SLOW_STEPS = 1; % Number of steps used during acceleration and slowwing down phase
	STEP_DURATION_VARIABILITY = .5; % Percentage of extra time over the average that can be used and still be in the same group of steps. Use a large value to include variations in speed that have very slow steps. Use .5 for normal walk and 1.9 or larger for varying speed; This condition is asociated to the plot with  title 'Stride duration';

	% Find the location of the maximun velocities for the opposite foot
	[aux,peaks_idx]=findpeaks(Vm,'minpeakheight',MIN_WALK_SPEED,'MINPEAKDISTANCE',MIN_FF_SEPARATION);
	% Designate max velocity points as likely footfalls for opossite foot
	FF_max_speed = zeros(size(Vm))';
	FF_max_speed(peaks_idx) = 1;
	FF_max_speed = (FF_max_speed == 1);

	% Determine the longest section with continious motion
	walking_ff_time = diff(peaks_idx);
	median_ff_time = median(walking_ff_time);
	walk_section = (walking_ff_time<median_ff_time*(1+STEP_DURATION_VARIABILITY));
	B=[0 walk_section 0];
	start_walks = find(diff(B)==1);
	end_walks = find(diff(B)==-1);
	walk_sizes = end_walks - start_walks;
	[aux,idx_walk] = max(walk_sizes);
	start_walk = start_walks(idx_walk);
	end_walk = end_walks(idx_walk);

	%Eliminate firs and last step, which may correspond to acceleration and slowing down
	start_walk = start_walk + ACCEL_SLOW_STEPS;
	end_walk = end_walk - ACCEL_SLOW_STEPS;

	if(PLOT_DETAILS) %Debugging plots
		figure,plot(walking_ff_time);hold on;
		plot(walk_section*median_ff_time,'ok');
		plot(start_walk,walking_ff_time(start_walk),'.g');
		plot(end_walk,walking_ff_time(end_walk),'.r');
		legend('Time between steps','Median walk time','Beggining of Walk','End of Walk');
		title('Stride duration');

		figure,plot(Vm);hold on;
		plot(peaks_idx,Vm(peaks_idx),'.k');
		plot(peaks_idx(start_walk),Vm(peaks_idx(start_walk)),'*g');
		plot(peaks_idx(end_walk),Vm(peaks_idx(end_walk)),'*r');
		legend('Speed','Max Speed','Beggining of Walk','End of Walk');
		title('Increase STEP_DURATION_VARIABILITY until it includes all the walking area');
	end;
	
	% Walking portion is defined at the point that the speed reaches WALK_SPEED_PERCENTAGE of the median speed value
	median_vel = mean(Vm(FF_max_speed));
	likely_walk_sections = find(Vm>median_vel*WALK_SPEED_PERCENTAGE);
	FF_stand_still_mask= zeros(size(FF_org));
	FF_stand_still_mask(1:likely_walk_sections(1)) = 1;
	FF_stand_still_mask(likely_walk_sections(end):end) = 1;
	
	FF_walking = zeros(size(FF_max_speed));
	FF_walking(peaks_idx(start_walk):peaks_idx(end_walk)) = FF_max_speed(peaks_idx(start_walk):peaks_idx(end_walk));
	FF_walking = FF_walking & ~FF_stand_still_mask;
	FF_walking = (FF_walking == 1);

	% Footfall detection based on velocities of the opposite shoe
	% Compute a footfall region
	switch(MERGE_MODE)
	case 'LARGE_SPEED_AND_ORG'
		% Uses large speed and original solutions combined
		FF = FF_org|FFv;
	case 'MAX_SPEED_AND_ORG'
		% Uses maximun speed and original solutions combined
		FF = FF_org|FF_max_speed;
	case 'MAX_SPEED_OR_ORG'
		% Uses maximun speed as the first option, if it is not available, it will use the original solution
		% Use original FFs when the person stands still even in the middle of the trial
		% force a FF an the begining and end of trial
		FF = (FF_stand_still_mask & FF_org) | FF_max_speed;
	case 'ORG_UNCOUPLED'
		% This method will use the same results as foot_fall.m
		FF  =  FF_org;
	end;

	if(PLOT_DETAILS)
		figure;
		plot(Vm);
		hold on;
		plot(find(FF),Vm(FF),'o','MarkerEdgeColor','k', 'MarkerFaceColor','k', 'MarkerSize',12)
		plot(find(FF_max_speed),Vm(FF_max_speed),'o','MarkerEdgeColor','g', 'MarkerFaceColor','g', 'MarkerSize',9);
		plot(find(FF_org),Vm(FF_org),'o','MarkerEdgeColor','r', 'MarkerFaceColor','r', 'MarkerSize',6);
		plot(find(FF_walking),Vm(FF_walking),'o','MarkerEdgeColor','m', 'MarkerFaceColor','y', 'MarkerSize',4);
		legend('V','Foot-fall','FF max speed','FF org','FF walk'); 
	end;
