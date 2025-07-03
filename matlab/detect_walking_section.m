% Author: Lauro Ojeda, 2011-2015
function [FF_walking, FF_max_speed] = detect_walking_section(walk_info,MIN_WALK_SPEED)
	WALK_SPEED_PERCENTAGE  =  0.90; % Determines the first and last foot fall
	if(~exist('MIN_WALK_SPEED','var'))
		MIN_WALK_SPEED = 2; % minimun foot velocity while walking
	end;
	FF = find(walk_info.FF);
	Vm = walk_info.Vm;

	% Determine the most likely walking portion based on the MIN_WALK_SPEED value 

	[aux,peaks_idx]=findpeaks(Vm,'minpeakheight',MIN_WALK_SPEED);
	FF_max_speed = zeros(size(Vm))';
	FF_max_speed(peaks_idx) = 1;
	FF_max_speed = (FF_max_speed == 1);


	median_vel = median(Vm(peaks_idx));
	likely_walk_sections = find(Vm>median_vel*WALK_SPEED_PERCENTAGE);
	footfall_index = find(FF>likely_walk_sections(1) & FF<likely_walk_sections(end));
	FF_walking = zeros(size(Vm));
	FF_walking(FF(footfall_index(1):footfall_index(end))) = 1;
	FF_walking = (FF_walking == 1);

	% Determine if there are potential missdetected footfalls based on the step separation
	step_separation = diff(find(FF_walking));
	median_step_separation = median(step_separation);
	missdetected_footfalls = abs(step_separation-median_step_separation)>median_step_separation/2;
	if(sum(missdetected_footfalls))
		disp(sprintf('There are potential missdetected footfalls %d',sum(missdetected_footfalls)));
	end;
