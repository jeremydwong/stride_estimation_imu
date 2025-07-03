% Author: Lauro Ojeda, 2011-2015
function stepData = stride_segmentation(walk_info,PERIOD,FILTER,OUTLIER_SECTION_SECONDS)
	if(~exist('FILTER','var') | isempty(FILTER))
		FILTER = 0;
	end;

	if(~exist('OUTLIER_SECTION_SECONDS','var') | isempty(OUTLIER_SECTION_SECONDS))
		OUTLIER_SECTION_SAMPLES= [];
	else
		number_of_sections = size(OUTLIER_SECTION_SECONDS,1);
		OUTLIER_SECTION_SAMPLES = [];
		for(ii=1:number_of_sections)
			OUTLIER_SECTION_SAMPLES =  [OUTLIER_SECTION_SAMPLES,(floor(OUTLIER_SECTION_SECONDS(ii,1)/PERIOD):floor(OUTLIER_SECTION_SECONDS(ii,2)/PERIOD))];
		end;
	end;

	MAX_FF_TIME = 2/PERIOD;
	FF = find(walk_info.FF_walking);
	%FF = find(walk_info.FF_max_speed); //Used this to add a wider range of speeds
	swing_start = FF(1:end-1);
	swing_finish = FF(2:end);
	swing_time = diff(FF);
	too_long = find(swing_time > MAX_FF_TIME);
	if(~isempty(too_long)) 
		disp(sprintf('Eliminted %d out of %d steps because of time duration is longer than %d',size(too_long,1), length(swing_start), MAX_FF_TIME));
	end;
	swing_start(too_long) = [];
	swing_finish(too_long) = [];

	stepData = get_steps(swing_start,swing_finish,walk_info,PERIOD,FILTER,OUTLIER_SECTION_SAMPLES); 


