function [Xr, Yr]= rotate_angle(X, Y, ang)
	Xr = X .* cos(ang) - Y .* sin(ang);
	Yr = X .* sin(ang) + Y .* cos(ang);

