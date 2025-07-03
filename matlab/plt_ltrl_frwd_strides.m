% Author: Lauro Ojeda, 2011-2015
function plt_ltrl_fwrd_strides(strides,CREATE_NEW_FIGURE,USE_SWING_ONLY)
	SHOW_ENDPOINTS = 1;
	SHOW_NUMBER = 0;
	QUICK_PLOT = 0;
	
	if(~exist('CREATE_NEW_FIGURE','var') | CREATE_NEW_FIGURE==1)
		figure;
	end;
	hold on;

	if(exist('USE_SWING_ONLY','var') & USE_SWING_ONLY)
		ltrl = strides.ltrl_swing;
	else
		ltrl = strides.ltrl;
	end;

	if(QUICK_PLOT)
		plot(ltrl,strides.frwd);
		if(SHOW_ENDPOINTS)
			plot(ltrl(end, :),strides.frwd(end, :),'ko','MarkerEdgeColor','k', 'MarkerSize',5);
		end;
	else
		colors=jet(size(ltrl,2));
		for(ii=1:size(ltrl,2))
			plot(ltrl(:,ii),strides.frwd(:,ii),'Color',colors(ii,:));
			if(SHOW_ENDPOINTS)
				plot(ltrl(end,ii),strides.frwd(end,ii),'ko','MarkerEdgeColor','k', 'MarkerFaceColor','k', 'MarkerSize',5);
			end;
			if(SHOW_NUMBER)
				text(ltrl(end,ii)+.001,strides.frwd(end,ii),num2str(ii));
			end;
		end;
	end;

	grid on;ylabel('Frwd [m]');xlabel('Ltrl [m]');
	hold off;

