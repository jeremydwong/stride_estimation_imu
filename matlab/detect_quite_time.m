% Author: Lauro Ojeda, 2003-2015
function static_period  =  detect_quite_time(W,PERIOD)
	PLOT_DETAILS  =  0;
	%Auto detect bias time
	MAXIMUM_STATIC_RATE = 2*pi/180; % [deg/sec]
	MINIMUN_STATIC_SAMPLES = 1.5/PERIOD; % Time converted into samples
	CONTINIOUS_SAMPLE_SPACE = 1;
	MAXIMUN_BIAS_SIGNAL_VAR = 2;

	Wm = (sum((W.^2)')).^.5/PERIOD; % It msut be multiplied by period to convert to [deg/sec]
	dWm = diff(Wm);
	low_dynamic_periods = find(abs(dWm) < MAXIMUM_STATIC_RATE);
	diff_low_dynamic_periods = diff(low_dynamic_periods);
	% Force a non static ponit at the beggining and at the end to mae sure that it includes this sections
	not_static = [1,find(diff_low_dynamic_periods > CONTINIOUS_SAMPLE_SPACE), length(diff_low_dynamic_periods)];
	diff_not_static = diff(not_static);
	most_likely_static_sections = find(diff_not_static > MINIMUN_STATIC_SAMPLES);
	if(isempty(most_likely_static_sections))
		warning('Did not find any suitable static signal, results may not be accurate');
		most_likely_static_sections = find(diff_not_static > MINIMUN_STATIC_SAMPLES/10);
	end;
	
	start_static_section = low_dynamic_periods(not_static(most_likely_static_sections));
	end_static_section = low_dynamic_periods(not_static(most_likely_static_sections + 1));

	static_period = [];
	for(i = 1:size(start_static_section,2))
			static_period = [static_period,start_static_section(i):end_static_section(i)];
	end;

	% find first order statistical analisis of bias signal to find the best static condition
	bias_var = std(Wm(static_period));
	bias_mean = median(Wm(static_period));
	best_static_signal = find(abs(Wm(static_period) - bias_mean) < bias_var*MAXIMUN_BIAS_SIGNAL_VAR);
	static_period = static_period(best_static_signal);

	if(PLOT_DETAILS)
		figure,plot(dWm);
		hold on;grid on;
		plot(low_dynamic_periods,dWm(low_dynamic_periods),'.g');
		plot(static_period,dWm(static_period),'or');
		plot(static_period,dWm(static_period),'.k');
	end;
