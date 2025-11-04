% Author: Lauro Ojeda, 2008-2015
function [W,A, static_period, M]  =  getData(W, A, PERIOD, varargin) %<------- added static period
% BIAS, is the static time at the beggining of the experiment
% leave BIAS the parameter empty to make the system detect the static time automatically

    plotFigures = 0;
    sectionSeconds = [];
    BIAS = [];
    M = [];
    if nargin >= 2 % optional arguments
        property_argin = varargin;
        % Step through the optional arguments
        while length(property_argin) >= 2
            prop = property_argin{1};
            val = property_argin{2};
            property_argin = property_argin(3:end);
            switch prop
              case 'PERIOD'
                PERIOD = val;
              case 'sectionSeconds'
                sectionSeconds = val;
              case 'BIAS'
                BIAS = val;
              case 'M'
                M = val;
              case 'plotFigures'
                plotFigures = val;
            end
        end
    end

   GRAVITY = 9.80297286843;
   if ~isempty(sectionSeconds)
        number_of_sections = size(sectionSeconds,1);
        SECTION_SAMPLES = [];
        for ii = 1:number_of_sections
            SECTION_SAMPLES  =  [SECTION_SAMPLES,(floor(sectionSeconds(ii,1)/PERIOD):floor(sectionSeconds(ii,2)/PERIOD))];
        end
        W = W(SECTION_SAMPLES,:);
        A = A(SECTION_SAMPLES,:);
        if ~isempty(M) 
            M = M(SECTION_SAMPLES,:);
        end
   end
    if ~isempty(BIAS)
        static_period = (1:BIAS/PERIOD);
    else
        static_period = detect_quite_time(W,PERIOD);
    end

    % Apply gyro static bias compensation
    W = W-ones(size(W,1),1)*mean(W(static_period,:));

    % Normalizes accelerometer during static time using a 1-G compensation
    static_acceleration = sum((A(static_period,:).^2)').^.5;
    gravity_measurement = mean(static_acceleration);
    A = A*GRAVITY/gravity_measurement;

    if ~isempty(M) 
        include_magnetomter = 1;
    else
        include_magnetomter = 0;
    end

    t = (1:size(W,1))*PERIOD;

    
%         f_cutoff = 3; 
%     N = 4;
%     frameRate_run = 1/PERIOD;
%     Wn = [2*f_cutoff/frameRate_run];
%     [B_Filt, A_Filt] = butter(N, Wn,'low');%'low'
%     W = filtfilt(B_Filt, A_Filt, W);
%     A = filtfilt(B_Filt, A_Filt, A);
%  
    
    if plotFigures
        figure,
        sh_xa(1) = subplot(2 + include_magnetomter,1,1);
        hold on;
        plot(t(static_period),W(static_period,:)/PERIOD*180/pi,'.k');
        legend('Static time');
        plot(t,W/PERIOD*180/pi);

        % plot(t,W(:,1)/PERIOD*180/pi, 'r');
        % plot(t,W(:,2)/PERIOD*180/pi, 'g');
        % plot(t,W(:,3)/PERIOD*180/pi, 'k');

        grid on;
        ylabel('W [deg/s]');
        title('Body referenced inertial signals');
        hold off;

        sh_xa(2) = subplot(2 + include_magnetomter,1,2);plot(t,A);
        grid on;ylabel('A [m/s^2]');

        if(include_magnetomter)
            sh_xa(3) = subplot(3,1,3);plot(t,M); grid on;ylabel('Mag');
        end;

        linkaxes(sh_xa,'x');
        xlabel('time [s]');
    end
