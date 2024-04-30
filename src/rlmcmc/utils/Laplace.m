function out = Laplace(varargin)
%
% Laplace(mu,Sigma) returns a sample from the multivariate Laplace
% distribution with mean mu and scale matrix Sigma
%
% Laplace(x,mu,Sigma) returns log(pdf(x)) for the Laplace distribution 
% with mean mu and scale matrix Sigma, up to an x-independent constant

if nargin == 2

    % arguments
    mu = varargin{1};
    Sigma = varargin{2};
    d = size(Sigma,1);

    % Cholesky factorisation of Sigma = R' * R
    R = chol(Sigma);

    % sample d times from standard univariate Laplace
    % standard Laplace == difference of two standard exponential RVs
    z = exprnd(1,d,1) - exprnd(1,d,1);

    % transform to correlated multivariate Laplace
    out = mu + (R') * z;

elseif nargin == 3

    % arguments
    x = varargin{1};
    mu = varargin{2};
    Sigma = varargin{3};

    % make sure x is a column vector
    x = x(:);

    % Cholesky factorisation of Sigma = R' * R
    R = chol(Sigma);

    % return log(pdf(x)) up to an x-independent constant
    out = - norm((R')\(x-mu),1);

end

end