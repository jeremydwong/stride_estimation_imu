function out = readLogomaticMemsenseFilev05(filename,varargin);
% function out = readLogomaticMemsenseFilev04(filename,varargin);

if nargin == 2
    maxnumberbytestoanalyze = varargin{1};
else
    maxnumberbytestoanalyze = Inf;
end

% OPTION to unwrap the Gyro Data or Not:
unwrapgyrodata = 0 ; 	% Set to 0 to just take data without unwrapping.

gravity = 9.806 ;
bytespersample = 38 ;

% Define Scale Factors
sampletimerscaleSecperbit = 2.1701e-6;
gyroscaleDegPerSecperbit = 5.4932e-2;
accelscaleGravperbit = 4.5776e-4;
magnetscaleGaussperbit = 8.6975e-5;
tempscaleCelsiusperbit = 1.8165e-2;

%% Read the File
fid = fopen(filename,'r');

% Find the first good packet delimiter
h = fread(fid,3*bytespersample,'*uchar',0,'ieee-be')';  % get a few bytes to find the first packet
inewpacket = strfind(h, char([255 255 255 255]) );
for ii = 1:length(inewpacket) ;
	if any(inewpacket - inewpacket(ii) == bytespersample)
	inewpacket = inewpacket(ii); 
	break;
	end
end
fprintf('Discarded %d bytes at start of file.\n',inewpacket-1) ;
% get packets in raw byte form
fseek(fid,inewpacket - 1,'bof');	% the "- 1" is to accommodate zero-based file addressing in fread function
data = fread(fid, [bytespersample,Inf] ,'*uint8', 0, 'ieee-be')' ;	% uint8
fclose(fid); 	% Close the file


%%%%%%%%%%%%%%%%%%% INTERPRET THE DATA %%%%%%%%%%%%%%%%%%%%%%
%% Mostly Copied from the older readLogomaticMemsenseFilev04 code
delimarray = data(:,1:4);
messagesizearray = typecast(data(:,5),'int8');
deviceIDarray = typecast(data(:,6),'int8');
messageIDarray = typecast(data(:,7),'int8');
sampletimerarray = bitor(bitshift(uint16(data(:,8)),8),bitshift(uint16(data(:,9)),0));
for kk = 10:13
    reservedarray(:,kk-9) = typecast(data(:,kk),'int8') ;
end
deviceid = bitor(bitshift(uint16(reservedarray(1,3)),8),bitshift(uint16(reservedarray(1,4)),0));
dataArray16 = bitor(bitshift(uint16(data(:,[14:2:37])),8),bitshift(uint16(data(:,[15:2:37])),0));
% dataArray = arrayfun(@(x) typecast(x,'int16'), dataArray16);
for kk = 1:size(dataArray16,2)
    dataArray(:,kk) = double(typecast(dataArray16(:,kk),'int16')) ;
end

checksumarray = data(:,38);

%% Compute CheckSums
badchecksums = find(checksumarray ~= bitand(hex2dec('FF'), sum(uint32(data(:,1:37)),2)));
if isempty(badchecksums)
	fprintf('No Bad Checksums\n')
else
	if length(badchecksums) == 1
		if badchecksums == size(data,1);
			display('Last Packet Bad Checksum - setting to NaN')
		end
	else
		fprintf('Some Bad Checksums - setting these rows to NaN:\n')
		fprintf('Row %8d  Checksum %8d  PacketSum %8d\n', [uint32(badchecksums) uint32(checksumarray(badchecksums)) uint32(bitand(hex2dec('FF'), sum(uint32(data(badchecksums,1:37)),2)))] );
	end
	dataArray(badchecksums,:) = NaN;
end

sampletimerData = double(sampletimerarray)*sampletimerscaleSecperbit;

accelData = dataArray(:,4:6)*accelscaleGravperbit;
magnetData = dataArray(:,7:9)*magnetscaleGaussperbit;
tempData = dataArray(:,10:12)*tempscaleCelsiusperbit;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% unwrap gyro data?
rawgyroData = dataArray(:,1:3); 
if unwrapgyrodata
	% gyroData = gyroscaleDegPerSecperbit * unwrap(rawgyroData*(pi)/(2^15),0.5*pi,1) *(2^15)/(pi) ;	% scale it to a +/-pi range, then use the stock "unwrap" function (along dimension 1) and scale it back 
	gyroData = gyroscaleDegPerSecperbit * dewrap(rawgyroData) ;	% special helper function below.
else
	gyroData = rawgyroData*gyroscaleDegPerSecperbit;
end

% put into a Structure. 
out.deviceid = deviceid ;
out.sampletimerData = sampletimerData;
out.gyroData = gyroData;
out.accelData = accelData;
out.magnetData = magnetData;
out.tempData = tempData;
out.rawbyteData = data;
% out.badDelims = baddelims;	% obsolete - no longer exists
% out.goods = goods;	% obsolete - no longer exists
% out.nfixes = sum(goods ~= 10);	% obsolete - no longer exists
out.time = (0:size(out.sampletimerData)-1)' / 150;

end % end main function


function out = dewrap(rawgyroData)
	correction = 2^16 ;
	spanOFtoSAT = 3640 ;
	thresholdBits = 10000 ;
	max16 = 32767 ;
	min16 = -32768 ;
	 
	for ii = 1:3
		fixloOF = 1;
		while(1)	% Loop: check if there are any points that make the big jump
		   hiOF = find( (rawgyroData(1:end-1,ii)> thresholdBits )  .*  (rawgyroData(2:end,ii) < (min16 + spanOFtoSAT ) ) ) + 1;
		   loOF = find( (rawgyroData(1:end-1,ii)< -thresholdBits)  .*  (rawgyroData(2:end,ii) > (max16 - spanOFtoSAT ) ) ) + 1;
		   if(isempty(hiOF)) 
				fixloOF = 1;
				if (isempty(loOF))
					fprintf('Component %d No OverFlows\n',ii);
					break; 
				end;
		   end
		   rawgyroData(hiOF,ii)=rawgyroData(hiOF,ii)+correction;
		   if fixloOF
		   	   rawgyroData(loOF,ii)=rawgyroData(loOF,ii)-correction;
		   end
		   fprintf('Component: %d   HighOverFlows: %d   LowOverFlows: %d\n',[ii size(hiOF,1) size(loOF,1)]);
		end; 
	end
	 
	 out = rawgyroData;
end	% end function "dewrap"

