% Author: Lauro Ojeda, 2011-2015
function plt_fwrd_elev_strides(strides,CREATE_NEW_FIGURE)
	SHOW_ENDPOINTS = 1;
	SHOW_NUMBER = 0;
	QUICK_PLOT = 0;
	ASSUME_FLAT_SURFACE = 1;
	
	if(~exist('CREATE_NEW_FIGURE','var') | CREATE_NEW_FIGURE==1)
		figure;
	end;
	hold on;

	if(QUICK_PLOT)
		plot(strides.frwd, strides.elev);
		if(SHOW_ENDPOINTS)
			plot(strides.frwd(end, :), strides.elev(end, :),'ko','MarkerEdgeColor','k', 'MarkerSize',5);
		end;
	else
		colors=jet(size(strides.ltrl,2));
		for(ii=1:size(strides.ltrl,2))
			if(ASSUME_FLAT_SURFACE)
				elev=strides.elev(:,ii); 
                delev=diff(elev); 
                NN=find(delev,1,'last');
                elev=elev(1:NN); err=elev(end);
                elev=elev-(1:NN)'/NN*err;
                elev(NN+1:length(strides.elev(:,ii))) = 0;
			else
				elev = strides.elev(:,ii);
			end;
            plot(-elev,'Color',colors(ii,:));
            5
			%plot(strides.frwd(:,ii),-elev,'Color',colors(ii,:));
			if(SHOW_ENDPOINTS)
				plot(strides.frwd(end,ii),-elev(end),'ko','MarkerEdgeColor','k', 'MarkerFaceColor','k', 'MarkerSize',5);
			end;
			if(SHOW_NUMBER)
				text(strides.frwd(end,ii)+.001,-elev(end),num2str(ii));
			end;
		end;
	end;
	grid on;ylabel('Elevation [m]');xlabel('Forward [m]');
	hold off;

