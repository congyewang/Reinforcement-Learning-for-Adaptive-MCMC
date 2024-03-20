function [res] = mixture_gaussian_target(x, weights, mu, var)

if nargin < 2
    weights = [0.5, 0.5];
end
if nargin < 3
    mu = [-3, 0; 3, 0];
end
if nargin < 4
    var = cat(3, eye(2), eye(2));
end

log_probs = zeros(size(weights));
for i = 1:length(weights)
    log_probs(i) = log(weights(i)) + logmvnpdf(x', mu(i,:), var(:,:,i));
end

res = logsumexp(log_probs);

end
