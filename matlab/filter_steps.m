% Author: Lauro Ojeda, 2011-2015
function [stride, number_of_steps] = filter_steps(stride, number_of_steps)
% Filter out steps that are not within known specifications
MAX_STEP_LENGHT = 1.8; % Used to eliminate very long stpes likely caused by non-detecetd footfalls
MIN_STEP_LENGHT = 0.5; % Used to eliminate very long stpes likely caused by non-detecetd footfalls
MAX_VAR = 2; % ELiminates outliers based on the variance from the median value
FILTER_ELEVATION = 0;

% Some of the filters use a double STD based filtering process 
% Eliminate long steps above the maximum limit
frwd = stride.frwd';
median_pos_frwd = median(frwd(:,end));
std_pos_frwd = std(frwd(:,end));
outlier = ( frwd(:,end)>MAX_STEP_LENGHT);
outlier = find(outlier);
if(~isempty(outlier))
	disp('_frwd LONG');
	[stride, number_of_steps] = cut_step_section(stride, outlier, number_of_steps);	
end;

% Eliminate very short steps
frwd = stride.frwd';
median_pos_frwd = median(frwd(:,end));
std_pos_frwd = std(frwd(:,end));
outlier = (frwd(:,end)<MIN_STEP_LENGHT);
outlier = find(outlier);
if(~isempty(outlier))
	disp('_frwd SHORT');
	[stride, number_of_steps] = cut_step_section(stride, outlier, number_of_steps);	
end;

% Eliminate steps away from the length median value
frwd = stride.frwd';
median_pos_frwd = median(frwd(:,end));
std_pos_frwd = std(frwd(:,end));
outlier = (abs(frwd(:,end)-median_pos_frwd)>std_pos_frwd*MAX_VAR);
outlier = find(outlier);
if(~isempty(outlier))
	disp('_frwd +2 VAR');
	[stride, number_of_steps] = cut_step_section(stride, outlier, number_of_steps);	
end;


% Eliminate steps that have too much side deviation
ltrl = stride.ltrl';
std_pos_ltrl = std(ltrl(:,end));
outlier = (abs(ltrl(:,end))>std_pos_ltrl*MAX_VAR);
outlier = find(outlier);
if(~isempty(outlier))
	disp('_ltrl +2 VAR');
	[stride, number_of_steps] = cut_step_section(stride, outlier, number_of_steps);	
end;


if(FILTER_ELEVATION)
	% Eliminate steps that have too much vertical deviation
	elev = stride.elev';
	median_pos_elev = median(elev(:,end));
	std_pos_elev = std(elev(:,end));
	outlier = (abs(elev(:,end)-median_pos_elev)>std_pos_elev*MAX_VAR);
	outlier = find(outlier);
	if(~isempty(outlier))
		disp('_elev +2 VAR steps');
		[stride, number_of_steps] = cut_step_section(stride, outlier, number_of_steps);	
	end;
end;

