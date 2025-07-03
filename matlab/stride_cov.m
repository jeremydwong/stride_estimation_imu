% Author: Lauro Ojeda, 2011-2015
function [cx,cy,ex,ey,covar] = stride_cov(x,y)
% Create a matrix containing all the data
data=[x;y]';
data_center = mean(data);
data=data-ones(size(data,1),1)*data_center;
sigma = 2;                     % standard deviations
% Computes the confidence value for a given sigma value 
% 68.3%, sigma=1; 95.4%, sigma=2; 99.7%, sigma=3 
confidence_value = diff(normcdf([-sigma sigma]));
% Compute the Mahalanobis radius of the ellipsoid that encloses
% the desired probability confidence 
scale = chi2inv(confidence_value,2);    
% Scale the covariance matrix so the confidence region has unit
% Mahalanobis distance.
covar = cov(data) * scale;
% The axes of the covariance ellipse are given by the eigenvectors of
% the covariance matrix.  Their lengths (for the ellipse with unit
% Mahalanobis radius) are given by the square roots of the
% corresponding eigenvalues.
[V D] = eig(covar);

% Create ellipse data points
t = linspace(0,2*pi,100);
e = [cos(t) ; sin(t)];        % unit circle
VV = V*sqrt(D);               % scale eigenvectors
e = VV*e;

ex=e(1,:);
ey=e(2,:);
cx=data(:,1);
cy=data(:,2);

