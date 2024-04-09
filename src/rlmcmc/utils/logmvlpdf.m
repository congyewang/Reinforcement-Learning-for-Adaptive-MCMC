function res = logmvlpdf(x, mu, Sigma)
% Ensure Mu and X are Column Vectors
% mu = mu(:);
% x = x(:);

% Dimension of Random Variable
[~,D] = size(x);

% Compute Terms
const = log(2) - (0.5 * D) * log(2 * pi);

% Compute Mahalanobis Distance
xc = bsxfun(@minus,x,mu);
mahala_dist = sqrt((xc / Sigma) * xc');

% Calculate Log PDF
res = const - 0.5 * logdet(Sigma) - mahala_dist;

end

function y = logdet(A)
U = chol(A);
y = 2*sum(log(diag(U)));
end
