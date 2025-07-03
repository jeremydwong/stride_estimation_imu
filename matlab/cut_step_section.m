% Author: Lauro Ojeda, 2012-2015
function [stride, number_of_steps] = cut_step_section(stride, out_of_bound, number_of_steps)
if(~isempty(out_of_bound))
	stride.frwd_swing(:,out_of_bound) = [];
	stride.ltrl_swing(:,out_of_bound) = [];
	stride.frwd(:,out_of_bound) = [];
	stride.ltrl(:,out_of_bound) = [];
	stride.abs_ltrl(:,out_of_bound) = [];
	stride.elev(:,out_of_bound) = [];
	stride.theta(:,out_of_bound) = [];
	stride.start_end(:,out_of_bound) = [];
	stride.foot_heading(out_of_bound) = [];
	stride.diff_foot_heading(out_of_bound) = [];
	stride.step_samples(out_of_bound) = [];
	stride.time(out_of_bound) = [];
	stride.frwd_speed_compensated(out_of_bound) = [];
	stride.frwd_speed(out_of_bound) = [];
	new_nember_of_steps = number_of_steps - length(out_of_bound);
	disp(sprintf('New number of steps %d out of %d', new_nember_of_steps, number_of_steps));
	number_of_steps = new_nember_of_steps;
end;



