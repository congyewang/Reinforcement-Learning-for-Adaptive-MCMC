function res = logmvlpdf(x, mu, Sigma)
% Ensure x is a matrix
if ~ismatrix(x)
    x = x(:).'; % Convert vector to row vector
end

% Ensure mu is a matrix
if ~ismatrix(mu)
    mu = repmat(mu, size(x, 1), 1); % Repeat mu to match the dimensions of x
end

% Set Sigma to identity matrix if it is missing
if nargin < 3 || isempty(Sigma)
    Sigma = eye(size(x, 2));
end

% Ensure Sigma is a matrix
if ~ismatrix(Sigma)
    Sigma = reshape(Sigma, size(x, 2), size(x, 2)); % Convert to square matrix
end

% Check if Sigma is symmetric positive definite
if ~isequal(Sigma, Sigma') || ~all(eig(Sigma) > 0)
    error('Matrix Sigma is not positive-definite.');
end

% Calculate the number of variables
k = size(Sigma, 1);

% Calculate ss
ss = x - mu;

% Calculate z
z = sum((ss / Sigma) .* ss, 2);
z(z == 0) = 1e-300; % Avoid log(0)

% Calculate log-determinant of Sigma
logdetSigma = logdet(Sigma);

% Calculate the density
res = log(2) - log(2 * pi) * (k / 2) - logdetSigma * 0.5 + ...
    (log(pi) - log(2) - log(2 * z) * 0.5) * 0.5 - ...
    sqrt(2 * z) - log(z / 2) * 0.5 * (k / 2 - 1);
end

function y = logdet(A)
U = chol(A);
y = 2*sum(log(diag(U)));
end
