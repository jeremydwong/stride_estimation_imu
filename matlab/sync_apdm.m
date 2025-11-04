% Author: Lauro Ojeda, 2012-2015
function [LeftWb,LeftAb,RightWb,RightAb,PERIOD,LeftMb,RightMb, static_period,varargout] = sync_apdm(L_FILE,R_FILE,varargin) %<--------------static_period

% function  [LeftWb, LeftAb, RightWb, RightAb, static_period, PERIOD, varargout] = syncApdm(L_FILE, R_FILE, varargin)

waistFlag = 0; headFlag = 0; handFlag = 0;

if nargin >= 2 % optional arguments
    property_argin = varargin;
    % Step through the optional arguments
    while length(property_argin) >= 2
        prop = property_argin{1};
        val = property_argin{2};
        property_argin = property_argin(3:end);
        switch prop
          case 'waist'
            W_FILE = val;
            if isempty(W_FILE)
                waistFlag = 0;
            else
                 waistFlag = 1;
            end
          case 'head'
            HEAD_FILE = val;
            if isempty(HEAD_FILE)
                headFlag = 0;
            else
                 headFlag = 1;
            end
          case 'hand'
            HAND_FILE = val;
            
            if isempty(HAND_FILE)
                handFlag = 0;
            else
                handFlag = 1;
            end
        end
    end
end

infoL = h5info(L_FILE);
L_time = h5read(L_FILE, [infoL.Groups(2).Groups(1).Name,'/Time']);

infoR = h5info(R_FILE);
R_time = h5read(R_FILE, [infoR.Groups(2).Groups(1).Name,'/Time']);

if headFlag
    infoHead = h5info(HEAD_FILE);
    Head_time = h5read(HEAD_FILE, [infoHead.Groups(2).Groups(1).Name,'/Time']);
end

if waistFlag
    infoW = h5info(W_FILE);
    W_time = h5read(W_FILE, [infoW.Groups(2).Groups(1).Name,'/Time']);
end

if handFlag
    infoHand = h5info(HAND_FILE);
    Hand_time = h5read(HAND_FILE, [infoHand.Groups(2).Groups(1).Name,'/Time']);
end

FREQ  = h5readatt(R_FILE, infoR.Groups(2).Groups(1).Groups(1).Name, 'Sample Rate');
PERIOD = 1/double(FREQ);

L_SECTION = [1,size(L_time,1)]*PERIOD;
R_SECTION = [1,size(R_time,1)]*PERIOD;

if waistFlag
    W_SECTION = [1,size(W_time,1)]*PERIOD;
end

if headFlag
    HEAD_SECTION = [1,size(Head_time,1)]*PERIOD;
end

if handFlag
    HAND_SECTION = [1,size(Hand_time,1)]*PERIOD;
end

if waistFlag && headFlag &&  handFlag
    [shift_val] = max([L_time(1) R_time(1) W_time(1) Head_time(1) Hand_time(1)]);
    shiftW = find(W_time >= shift_val,1);
    shiftHead = find(Head_time >= shift_val,1);
    shiftHand = find(Hand_time >= shift_val,1);  
elseif waistFlag && headFlag
    [shift_val] = max([L_time(1) R_time(1) W_time(1) Head_time(1)]);
    shiftW = find(W_time >= shift_val,1);
    shiftHead = find(Head_time >= shift_val,1);
elseif waistFlag && handFlag
    [shift_val] = max([L_time(1) R_time(1) W_time(1) Hand_time(1)]);
    shiftW = find(W_time >= shift_val,1);
    shiftHand = find(Hand_time >= shift_val,1);
elseif headFlag &&  handFlag
    [shift_val] = max([L_time(1) R_time(1) Head_time(1) Hand_time(1)]);
    shiftHead = find(Head_time >= shift_val,1);
    shiftHand = find(Hand_time >= shift_val,1);   
elseif handFlag
    [shift_val] = max([L_time(1) R_time(1) Hand_time(1)]);
    shiftHand = find(Hand_time >= shift_val,1);
elseif headFlag
    [shift_val] = max([L_time(1) R_time(1) Head_time(1)]);
    shiftHead = find(Head_time >= shift_val,1);
elseif waistFlag
    [shift_val] = max([L_time(1) R_time(1) W_time(1)]);
    shiftW = find(W_time >= shift_val,1);
else
    [shift_val] = max([L_time(1) R_time(1)]);
end

shiftL = find(L_time >= shift_val,1);
shiftR = find(R_time >= shift_val,1);

L_SECTION = [shiftL, size(L_time,1)]*PERIOD;
R_SECTION = [shiftR, size(R_time,1)]*PERIOD;


if waistFlag
    W_SECTION = [shiftW, size(W_time,1)]*PERIOD;
end

if handFlag
    HAND_SECTION = [shiftHand, size(Hand_time,1)]*PERIOD;
end

if headFlag
    HEAD_SECTION = [shiftHead, size(Head_time,1)]*PERIOD;
end 

% Load data 
[Wb,Ab,PERIOD,Mb] = getdata_apdm(L_FILE, 1);
[LeftWb,LeftAb, static_period, LeftMb] = getdata(Wb,Ab,PERIOD, 'sectionSeconds', L_SECTION, 'plotFigures', 1); %<--------------static_period
set(gcf,'Name','Left IMU');

[Wb,Ab,PERIOD,Mb] = getdata_apdm(R_FILE,1);
[RightWb,RightAb, ~, RightMb] = getdata(Wb,Ab,PERIOD, 'sectionSeconds', R_SECTION, 'plotFigures', 1); %<--------------static_period
set(gcf,'Name','Right IMU');

if headFlag 
    [Wb,Ab,PERIOD,Mb] = getdata_apdm(HEAD_FILE, 1);
    [HeadWb, HeadAb, ~, HeadMb] = getdata(Wb, Ab, PERIOD, 'sectionSeconds', HEAD_SECTION, 'plotFigures', 1); 
    set(gcf,'Name','Head IMU');
    varargout{1} = HeadWb;
    varargout{2} = HeadAb;
end

if waistFlag
    [Wb,Ab,PERIOD,Mb] = getdata_apdm(W_FILE,1);
    [WaistWb, WaistAb, ~, WaistMb] = getdata(Wb, Ab, PERIOD, 'sectionSeconds', W_SECTION, 'plotFigures', 1); 
    set(gcf,'Name','Waist IMU');
    varargout{3} = WaistWb;
    varargout{4} = WaistAb;
else
        varargout{3} = [];
    varargout{4} = [];
end


if handFlag
    [Wb,Ab,PERIOD,Mb] = getdata_apdm(HAND_FILE, 1);
    [HandWb, HandAb, ~, HandMb] = getdata(Wb, Ab, PERIOD,  'sectionSeconds', HAND_SECTION, 'plotFigures', 1); 
    set(gcf,'Name','Hand IMU');
    varargout{5} = HandWb;
    varargout{6} = HandAb;
else
        varargout{5} = [];
    varargout{6} = [];
end

end

