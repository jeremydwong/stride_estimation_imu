% Author: Lauro Ojeda, 2001-2015
function [accel_phi,accel_theta] = acc_tilt(A)
GRAVITY = 9.80297286843;
%Compute tilt form acceleration
AX = A(:,1);
AY = A(:,2);

%Compute pitch 
accel_theta = (AX)/GRAVITY;
valid_angles = find(abs(accel_theta) <= 1);
accel_theta(valid_angles) = asin(accel_theta(valid_angles));

%Compute roll angle
accel_phi = (AY)./cos(accel_theta)/GRAVITY;
valid_angles = find(abs(accel_phi) <= 1);
accel_phi(valid_angles) = -asin(accel_phi(valid_angles));

