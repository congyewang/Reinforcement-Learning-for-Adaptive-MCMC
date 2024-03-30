function [store, acc] = mala(nits, theta_start, epsilon)
if nargin < 1
    % number of MCMC iterations
    nits = 5000;
end
if nargin < 2
    % initial state for MCMC
    theta_start = [0, 0];
end
if nargin < 3
    epsilon = 1;
end

dim = size(theta_start, 2);
store = zeros(nits+1, dim);
current_theta = theta_start;
store(1, :) = current_theta;
nacc = 0;

% proposal mean
mean_func = @(x) x + epsilon * grad_log_target_pdf(x);
% proposal variance
variance = 2 * epsilon * eye(dim);

log_proposal_pdf = @(x, y) logmvnpdf(y, mean_func(x), variance);
% log acceptance ratio (x -> y)
log_alpha = @(x,y) log_target_pdf(y) ...
    - log_target_pdf(x) ...
    + log_proposal_pdf(y,x) ...
    - log_proposal_pdf(x,y);

for i = 1:nits
    proposed_theta = mvnrnd(mean_func(current_theta), variance);
    % Accepted / Rejected Process
    if log(rand()) < log_alpha(current_theta, proposed_theta)
        store(i+1, :) = proposed_theta;
        nacc = nacc + 1;

        current_theta = proposed_theta;
    else
        store(i+1, :) = current_theta;
    end

end

acc = nacc / nits;

end

function y = log_target_pdf(x)
y = logmvnpdf(x, 0, 1);
end

function dydx = grad_log_target_pdf(x)
dydx = -x;
end
