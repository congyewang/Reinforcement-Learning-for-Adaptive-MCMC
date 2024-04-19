function R = mvlaprnd(d,MU,SIGMA)
% Purpose:  
%       Generate multivariate Laplace random numbers
%
% Inputs: 
%       d       - dimension of the random vector 
%       MU      - mean vector
%       SIGMA   - covariance matrix
%
% Output:
%       R - Multivariate Laplacian random number
% 
% Reference:
%       T. Eltoft et. al.
%       "On the Multivariate Laplace Distribution"
%       IEEE Signal Processing Letters, Vol. 13, No. 5, May 2006
% 
% Remarks:
%       1. Should have installed sqrtm.m
%       2. Require exprnd.m and mvnrnd.m
% 
% Author:
%       Arun Muraleedharan (arunm_367@yahoo.co.in)
%       Graduate Researcher, TU Delft.
%       February 2017
% 
% Last Updated:
%       6th February, 2017
%
%
%
% Illustrative Example
% 
% d       = 2;
% MU      = zeros(d,1);
% SIGMA   = 0.5*eye(d);
% N       = 5000;
% R       = zeros(d,N);
% for i = 1:N
%     R(:,i) = mvlaprnd(d,MU,SIGMA);
% end
% 
% figure
% nbins = 100;
% hist3(R',[nbins nbins])
% title('2D Laplace Random Numbers')
% xlabel('X'); ylabel('Y'); zlabel('Count')
% set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
% grid on

if nargin ~= 3
    error('Input must have 3 arguments, neither more nor less!');
end

if ~all(eig(SIGMA)> 0)
    error('SIGMA is not positive definite')
end

lambda  = det(SIGMA);
GAMMA   = SIGMA/lambda;
z       = exprnd(lambda);
MU_G    = zeros(d,1); 
SIGMA_G = eye(d);
X       = mvnrnd(MU_G,SIGMA_G)';
R       = MU + sqrt(z)*sqrtm(GAMMA)*X;
end






