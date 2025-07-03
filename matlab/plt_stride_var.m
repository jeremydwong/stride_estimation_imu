% Author: Lauro Ojeda, 2011-2015
function plt_stride_var(strides,CREATE_NEW_FIGURE,DETREND_STEP_LENGTH, SHIFT_TO_ZERO)
	SHOW_NUMBER = 0;
	if(~exist('CREATE_NEW_FIGURE','var') | CREATE_NEW_FIGURE==1)
		figure;
	end;
	if(~exist('SHIFT_TO_ZERO','var'))
		SHIFT_TO_ZERO = 1;
	end;
	hold on;


	if(~exist('DETREND_STEP_LENGTH','var') | DETREND_STEP_LENGTH == 0)
		% Use full data
		frwd = strides.frwd(end,:)
	else
		% Use de-trended data
		frwd = strides.frwd_speed_compensated;
	end;

	ltrl = strides.ltrl(end,:);
	[cx,cy,ex,ey,covar] = compute_cov(ltrl,frwd, SHIFT_TO_ZERO);

	colors=jet(size(ltrl,2));
	for(ii=1:size(ltrl,2))
		plot(cx(ii),cy(ii),'o','MarkerEdgeColor','k', 'MarkerFaceColor',colors(ii,:), 'MarkerSize',5);
		if(SHOW_NUMBER)
			text(cx(ii)+.001,cy(ii),num2str(ii));
		end;
	end;
	plot(ex, ey, 'Color','r','LineWidth',3);
	grid on;ylabel('Frwd [m]');xlabel('Ltrl [m]');
	axis equal;
	hold off;

