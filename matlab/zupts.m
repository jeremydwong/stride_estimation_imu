% Author: Lauro Ojeda, 2003-2015
% Zero velocity Updates on accelerometer values
% It assumes that the error is linear over the step interval
if(i == 2)
	last_footfall = 0;
end;

%Applies ZUPTs on foot falls only
if(FF(i))
	step_range = (last_footfall+1:i);
	%Skip short duration FF
	if(size(step_range,2)<2)
		return;
	end;
	step_samples = size(step_range,2);
	%Compute final error
	velocity_error = sum(An(step_range,:));
	%Compute the accelerometer error assuming it is linear
	acceleration_error = velocity_error/step_samples;
	%Apply error corrections in accelerations
	Anz(step_range,:) = An(step_range,:) - ones(step_samples,1)*acceleration_error;
	last_footfall = i;
end;

