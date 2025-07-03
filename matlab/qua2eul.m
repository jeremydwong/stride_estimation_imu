% Author: Lauro Ojeda, 2003-2015
% Bibliography:
% David Titterton and John Weston, Strapdown Inertial Navigation Technology, 2nd Edition
% pp 45
% Converts quaternion to Euler representation
function euler = qua2eul(quaternion)
a = quaternion(:,1);
b = quaternion(:,2);
c = quaternion(:,3);
d = quaternion(:,4);

phi = atan2((2*(a.*b + c.*d)),(a.^2 - b.^2 - c.^2 + d.^2));
the = asin(2*(a.*c - d.*b));
psi = atan2((2*(a.*d + b.*c)),(a.^2 + b.^2 - c.^2 - d.^2));
euler = [phi,the,psi];
