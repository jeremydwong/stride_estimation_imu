% Author: Lauro Ojeda, 2003-2015
% Bibliography:
% - Ken Shoemake: quaternion_uaternions
% pp 7
% Given a rotationation expressed in quaternion format
% R*quaternion = rotationation about the X, Y and Z navigation axis
% quaternion*R = rotationation about the X, Y and Z body axis
% - David Titterton and John Weston, Strapdown Inertial Navigation Technology, 2nd Edition
% pp 320
function quaternion = qua_est(W,quaternion_prev)
mag = (sum(W.^2)).^.5;
if(mag ~= 0) 
	sin_mag = sin(mag/2.0)/mag;
else
	sin_mag = 0.5;
end;
rotation = [cos(mag/2.0);sin_mag*W'];
a = quaternion_prev(1); b = quaternion_prev(2); c = quaternion_prev(3); d = quaternion_prev(4);
quaternion_sqw = [a,-b,-c,-d;
		 		b, a,-d, c;
		 		c, d, a,-b;
		 		d,-c, b, a];

quaternion = quaternion_sqw*rotation;
