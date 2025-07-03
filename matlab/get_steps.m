% Author: Lauro Ojeda, 2011-2015
function [result] = get_steps(step_start,step_end,walk_info,PERIOD,FILTER,OUTLIER_SECTION)
	PLOT_DETAILS = 0;
	% DIRECTION_STEPS, determine the number of steps before and after current one, used to define a straight segment, this should >= 3 for best results
	DIRECTION_STEPS = 3;% default3;
%     mean_step_direction = 0;%---osman when the DIRECTION_STEPS I amde mean_step_direction zero so the code works COMMENT if  DIRECTION_STEPS non zero, default 3
    
	EXTRA_FRWD_CORRECTION = 1;%default is 1. 31 Oct2022       1; % This will straighten the paths perfectly
%     EXTRA_FRWD_CORRECTION = 0;%for hand reaching Oct 2023

	step_start = step_start(:);
	step_end = step_end(:);
	number_of_steps = size(step_start,1);
	longest_step = max(step_end - step_start) + 1;
	P = walk_info.P;
	euler = walk_info.euler;
	Vm = walk_info.Vm;

	% Create matrices to store results
	ltrl_swing = zeros(number_of_steps,longest_step);
	frwd_swing = zeros(number_of_steps,longest_step);
	ltrl = zeros(number_of_steps,longest_step);
	frwd = zeros(number_of_steps,longest_step);
	ltrl_straighten = zeros(number_of_steps, longest_step);
	frwd_straighten = zeros(number_of_steps, longest_step);
	abs_ltrl = zeros(number_of_steps,longest_step);
	abs_frwd = zeros(number_of_steps,longest_step);
	elev = zeros(number_of_steps,longest_step);
	theta = zeros(number_of_steps,longest_step);
	foot_heading = zeros(number_of_steps,1);
	diff_foot_heading = zeros(number_of_steps,1);
	start_end = zeros(number_of_steps,2);

	% Compute individual step direction
	direction = atan2(P(step_end,2) - P(step_start,2),P(step_end,1) - P(step_start,1));
	if(PLOT_DETAILS)
		% This is the average walk direction that is used to rotate the trajectory, is valid for short walks,
		% and when using low drift gltrl_pol_rotos
		Px =  P(step_start, 1);
		Py =  P(step_start, 2);
		pol = polyfit(Px, Py, 1);
		% Use the atan2 to determine the right grid quadrant
		overall_step_direction = atan2(polyval(pol, Px(end)) - polyval(pol, Px(1)), Px(end) - Px(1));
		[frwd_pol_rot, ltrl_pol_rot] = rotate_angle(Px, Py, -overall_step_direction);
		PATH_FIG = figure;
		hold on; grid on;
		plot(frwd_pol_rot, ltrl_pol_rot, 'k');
	end;

	% Unwrap the euler in order to eliminate discontinuities
	walk_foot_heading = unwrap(euler(step_end,3));
	% Perform a defaul line fit correction for heading
	x = (1:length(walk_foot_heading))';
	y = walk_foot_heading;
	pol = polyfit(x,y,1);
	heading_correction = polyval(pol,x);
	corrected_heading = y - heading_correction;
	if(PLOT_DETAILS)
		ANG_FIG = figure;
		plot(walk_foot_heading-pol(2));
		hold on; grid on;
		plot(heading_correction-pol(2),'r');
		plot(y-heading_correction,'g');
	end;
	
	for(i = 1:number_of_steps)
		frwd_swing(i,1:step_end(i) - step_start(i) + 1) = P(step_start(i):step_end(i),1)*cos(-direction(i)) - P(step_start(i):step_end(i),2).*sin(-direction(i));
		frwd_swing(i,step_end(i) - step_start(i) + 2:end) = frwd_swing(i,step_end(i) - step_start(i) + 1);
		ltrl_swing(i,1:step_end(i) - step_start(i) + 1) = P(step_start(i):step_end(i),1)*sin(-direction(i)) + P(step_start(i):step_end(i),2).*cos(-direction(i));
		ltrl_swing(i,step_end(i) - step_start(i) + 2:end) = ltrl_swing(i,step_end(i) - step_start(i) + 1);

		% Uses the nearby steps to determine the angle, is less sensitive to gltrl_pol_roto drift
		% if a number oif DIRECTION_STEPS is defined, the mean_step_direction value is redefined
		if(DIRECTION_STEPS)
			% Select the steps +/- DIRECTION_STEPS
			if(i>DIRECTION_STEPS & number_of_steps - i>DIRECTION_STEPS)
				nearby_steps_index = (i - DIRECTION_STEPS:i + DIRECTION_STEPS);
			elseif(i <= DIRECTION_STEPS)
				nearby_steps_index = (1:i + DIRECTION_STEPS);
			else
				nearby_steps_index = (i - DIRECTION_STEPS:length(step_start));
			end;
			% Find local direction of travel
			nearby_steps = step_start(nearby_steps_index);
			x = P(nearby_steps, 1);
			y = P(nearby_steps, 2);
			pol = polyfit(x, y, 1);
			mean_step_direction = atan2(polyval(pol, x(end)) - polyval(pol, x(1)), x(end) - x(1));
			% Find a local heading correction
			y = walk_foot_heading(nearby_steps_index);
			x = (nearby_steps_index)';
			pol = polyfit(x, y, 1);
			heading_correction = polyval(pol, x);
			current_index = find(x==i);
			corrected_heading(i) = y(current_index) - heading_correction(current_index);
		end;
			
		[frwd(i,1:step_end(i) - step_start(i) + 1), ltrl(i,1:step_end(i) - step_start(i) + 1)] = rotate_angle(P(step_start(i):step_end(i),1), P(step_start(i):step_end(i),2), -mean_step_direction);
		ltrl(i,step_end(i) - step_start(i) + 2:end) = ltrl(i,step_end(i) - step_start(i) + 1);
		frwd(i,step_end(i) - step_start(i) + 2:end) = frwd(i,step_end(i) - step_start(i) + 1);
		if(i>1)
			ltrl(i,:) = ltrl(i,:) - ltrl(i,1) + ltrl(i-1,end);
		else
			ltrl(i,:) = ltrl(i,:) - ltrl(i,1);
		end;
		% Store elevation information
		elev(i,1:step_end(i) - step_start(i) + 1) = P(step_start(i):step_end(i),3);
		elev(i,step_end(i) - step_start(i) + 2:end) = elev(i,step_end(i) - step_start(i) + 1);
		% Store pitch angle
		theta(i,1:step_end(i) - step_start(i) + 1) = euler(step_start(i):step_end(i),2);
		theta(i,step_end(i) - step_start(i) + 2:end) = theta(i,step_end(i) - step_start(i) + 1);
		step_samples(i) = step_end(i) - step_start(i);
		% Store heading angle
		foot_heading(i) = corrected_heading(i);
		diff_foot_heading(i) =  euler(step_end(i),3)-euler(step_start(i),3); 
		start_end(i,:) = [step_start(i), step_end(i)]; 
	end;

	ltrl_end = ltrl(: , end);
	frwd_end = frwd(: , end);
	pol = polyfit(frwd_end, ltrl_end, 1);
	frwd_pol = (min(frwd_end) : max(frwd_end));
	ltrl_pol = polyval(pol, frwd_pol);

	if(PLOT_DETAILS)
		figure(ANG_FIG);
		hold on;
		plot(foot_heading,'k');
		xlabel('Step #');ylabel('Ang [rad]');
		legend('Org','Linear Fit','Line-Fit Correction','Piecewise Correction');
		hold off;

		figure(PATH_FIG)
		hold on;
		plot(frwd(:,end), ltrl(:,end),'b');
		xlabel('X [m]');ylabel('Y [m]');
		plot(frwd_end, ltrl_end, '*');
		plot(frwd_pol, ltrl_pol, 'b');
		hold off;
		legend('Line-Fit Correction','Piecewise Correction', '', '');
	end;

	if(EXTRA_FRWD_CORRECTION)
		mean_step_direction =  atan(pol(1));
		[frwd_pol_rot, ltrl_pol_rot] = rotate_angle(frwd_pol, ltrl_pol, -mean_step_direction);
		ltrl_pol_rot = ltrl_pol_rot - mean(ltrl_pol_rot);
		[frwd_end, ltrl_endr] = rotate_angle(frwd_end, ltrl_end, -mean_step_direction);
		center_ltrl_end =  mean(ltrl_endr);
		ltrl_endr = ltrl_endr - center_ltrl_end;

		for(i = 1 : number_of_steps)
			[frwd_straighten(i, :), ltrl_straighten(i, :)] = rotate_angle(frwd(i, :), ltrl(i, :), -mean_step_direction);
			ltrl_straighten(i, :) = ltrl_straighten(i, :) - center_ltrl_end;
		end;

		ltrl = ltrl_straighten;
		frwd = frwd_straighten;

		if(PLOT_DETAILS)
			hold on;
			plot(frwd_pol_rot, ltrl_pol_rot, 'g');
			plot(frwd_end, ltrl_endr, 'g*');
			plot(frwd_straighten(:,end),ltrl_straighten(:,end),'g');
			xlabel('X [m]');ylabel('Y [m]');
			legend('Line-Fit Correction','Piecewise Correction', '', '', 'Best correction');
			hold off;
		end;
	end;

	% Translate foot fall location to the origin
	frwd_swing = frwd_swing - frwd_swing(:,1)*ones(1,longest_step);
	abs_frwd = frwd;
	frwd = frwd - frwd(:,1)*ones(1,longest_step);
	ltrl_swing = ltrl_swing - ltrl_swing(:,1)*ones(1,longest_step);
	abs_ltrl = ltrl;
	ltrl = ltrl - ltrl(:,1)*ones(1,longest_step);
	elev = elev - elev(:,1)*ones(1,longest_step);
	%theta = theta - theta(:,1)*ones(1,longest_step);

%     %osman THIS IS NOT A PERMANET SOLUTION
%     if ~isempty(find(frwd(:,end)< 0))  
%         frwd(:,end) = abs(frwd(:,end));
%     end
    
    if ~isempty(find(frwd(:,end)< 0))  
%         keyboard();
%         error('in get steps');
        warning('OSMAN NEGATIVE IN FRWRD')
        frwd = abs(frwd);
    end
     
	% Compute step speed
	step_length = frwd(:,end)'; 
	time = step_samples*PERIOD;
	step_speed = step_length./time; 
	pol = polyfit(step_speed,step_length,1); step_length_fit = polyval(pol,step_speed);
	frwd_speed_compensated = step_length - step_length_fit;
    frwd_speed = step_speed; 
    
    
	% Compute first order statistics and assemble result structure
	result.frwd_swing = frwd_swing'; 
	result.ltrl_swing = ltrl_swing';
	result.frwd = frwd'; 
	result.ltrl = ltrl';
	result.abs_ltrl = abs_ltrl';
	result.abs_frwd = abs_frwd';
	result.elev = elev';
	result.theta = theta';
	result.start_end = start_end';
	result.foot_heading = foot_heading;
	result.diff_foot_heading = diff_foot_heading;
	result.step_samples = step_samples; 
	result.time = time; 
	result.frwd_speed_compensated = frwd_speed_compensated; 
	result.frwd_speed = frwd_speed; 

	% Eliminate user defined ouliers
	out_of_bound=[];
	for(i = 1:number_of_steps)
		if(sum(step_start(i)==OUTLIER_SECTION) | sum(step_end(i)==OUTLIER_SECTION))
			out_of_bound = [out_of_bound,i];
		end;
	end;
	
	if(~isempty(out_of_bound))
		disp('User defined outliers');
		[result, number_of_steps] = cut_step_section(result, out_of_bound, number_of_steps);
	end;

	if(PLOT_DETAILS)
		t = (1:length(Vm))*PERIOD;
		figure;
		plot(t,Vm);hold on;
		plot(t(OUTLIER_SECTION),Vm(OUTLIER_SECTION),'.y');
		plot(t(step_start(1)),Vm(step_start(1)),'*g');
		plot(t(step_end(end)),Vm(step_end(end)),'*r');
		xlabel('Sample #');
		ylabel('Speed [m/sec]');
		legend('Vm','Start','End');
		grid on;
	end;
	
	if(FILTER)
		[result, number_of_steps] = filter_steps (result, number_of_steps);
	end;


