% Author: Lauro Ojeda, 2011-2015
function plt_strides(stepLeft,stepRight)
	PLT_ELEV = 1 ;
	%% Plot results 
	LCOL=jet(size(stepLeft.mPosLtrl,2));
	RCOL=jet(size(stepRight.mPosLtrl,2));

	figure,
	subplot(1,2,1), hold on;
	for(ii=1:size(stepLeft.mPosLtrl,2))
		plot(stepLeft.mPosLtrl(:,ii),stepLeft.mPosFrwd(:,ii),'Color',LCOL(ii,:));
	end;
	grid on;ylabel('Frwd [m]');xlabel('Left Ltrl [m]');
	hold off;
	AX1=axis;

	subplot(1,2,2), hold on;
	for(ii=1:size(stepRight.mPosLtrl,2))
		plot(stepRight.mPosLtrl(:,ii),stepRight.mPosFrwd(:,ii),'Color',LCOL(ii,:));
	end;
	grid on;ylabel('Frwd [m]');xlabel('Right Ltrl [m]');
	hold off;
	AX2=axis;
	AX=[min([AX1(1),AX2(1)]),max([AX1(2),AX2(2)]),0,max([AX1(4),AX2(4)])];
	axis(AX); subplot(1,2,1);axis(AX);

	if(PLT_ELEV)
		figure,
		subplot(2,1,1), hold on;
		for(ii=1:size(stepLeft.mPosLtrl,2))
			plot(stepLeft.mPosFrwd(:,ii),-stepLeft.mPosElev(:,ii),'Color',LCOL(ii,:));
		end;
		grid on;ylabel('Left Elev [m]');xlabel('Frwd [m]');
		hold off;
		AX1=axis;
		subplot(2,1,2), hold on;
		for(ii=1:size(stepRight.mPosLtrl,2))
			plot(stepRight.mPosFrwd(:,ii),-stepRight.mPosElev(:,ii),'Color',LCOL(ii,:));
		end;
		grid on;ylabel('Right Elev [m]');xlabel('Frwd [m]');
		hold off;
		AX2=axis;
		AX=[0,max([AX1(2),AX2(2)]),min([AX1(3),AX2(3)]),max([AX1(4),AX2(4)])];
		axis(AX); subplot(2,1,1);axis(AX);

	end;
