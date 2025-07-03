% Before running this tool, load both files using:
% 	[Wb1,Ab1,Wb2,Ab2,PERIOD] = sync_apdm(L_FILE,R_FILE,0);
% 	and call it as find_data_shift(Wb1,Wb2,PERIOD)
function [shift] = find_data_shift(lW,rW,PERIOD)
% Equalize sizes
FREQ = 1/PERIOD;
lW= lW(:,1);
rW= rW(:,1);
lN=size(lW,1);
rN=size(rW,1);
if(lN<rN)
	rW=rW(1:lN);
	N=lN;
else
	lW=lW(1:rN);
	N=rN;
end;

% Find maximun signal correlation
sng_corr=xcorr(lW,rW);
[aux,max_corr]=max(sng_corr);
[aux,bef_max]=max(sng_corr(1:max_corr-FREQ/2));
[aux,aft_max]=max(sng_corr(max_corr+FREQ/2:end));
aft_max=aft_max+max_corr+FREQ/2-1;

% Compute step size
step_dist=round(mean(diff([bef_max, max_corr, aft_max])));

% Compute signal shift
shift=round(N-max_corr+1-step_dist/2);
if(shift<0)
	disp('Shift Left');
else
	disp('Shift Right');
end;
disp(shift);

