% Author: Lauro Ojeda, 2003-2015
% Bibliography:
% - David Titterton and John Weston, Strapdown Inertial Navigation Technology, 2nd Edition
% pp 44
% - Ken Shoemake: Quaternions 
% rotation_matrix = LQ*RQ page 6 
% This can be multiplied by a vector V in order to rotate it
% QVQ' = LQ*RQ*V
% The last column and rows can be ignored because they are either 0 or 1 and are not used 
% to rotate vectors
function [rotation_matrix] = qua2rot(quaternion)
a = quaternion(1);
b = quaternion(2);
c = quaternion(3);
d = quaternion(4);
rotation_matrix(1,1) = a^2 + b^2 - c^2 - d^2;
rotation_matrix(1,2) = 2*(b*c - a*d);
rotation_matrix(1,3) = 2*(b*d + a*c);

rotation_matrix(2,1) = 2*(b*c + a*d);
rotation_matrix(2,2) = a^2 - b^2 + c^2 - d^2;
rotation_matrix(2,3) = 2*(c*d - a*b);

rotation_matrix(3,1) = 2*(b*d - a*c);
rotation_matrix(3,2) = 2*(c*d + a*b);
rotation_matrix(3,3) = a^2 - b^2 - c^2 + d^2;
