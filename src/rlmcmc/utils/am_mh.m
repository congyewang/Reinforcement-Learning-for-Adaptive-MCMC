function [store, acc] = am_mh( ...
    nits, ...
    theta_start, ...
    initial_mu, ...
    initial_Sigma, ...
    lambda_initial, ...
    gamma_sequence, ...
    alpha_star ...
    )
%{
     AM algorithm with global adaptive scaling.
%}

if nargin < 1 || isempty(nits)
    nits = 5000;
end
if nargin < 1 || isempty(theta_start)
    theta_start = [0, 0];
end

sample_dim = size(theta_start, 2);

if nargin < 3 || isempty(initial_mu)
    initial_mu = zeros(1, sample_dim);
end
if nargin < 4 || isempty(initial_Sigma)
    initial_Sigma = eye(sample_dim);
end
if nargin < 5 || isempty(lambda_initial)
    lambda_initial = 1;
end
if nargin < 6 || isempty(gamma_sequence)
    gamma_sequence = arrayfun(@(i) 1 / (i + 1)^0.6, 0:(nits-1));
end
if nargin < 7 || isempty(alpha_star)
    alpha_star = 0.234;
end

% Initialize
theta = zeros(nits + 1, sample_dim);
mu = zeros(nits + 1, sample_dim);
Sigma = zeros(sample_dim, sample_dim, nits + 1);
lambda_seq = zeros(1, nits + 1);

theta(1, :) = theta_start;
mu(1, :) = initial_mu;
Sigma(:, :, 1) = initial_Sigma;
lambda_seq(1) = lambda_initial;

nacc = 0;

for i = 1:nits
    % Sample from proposal distribution
    theta_prop = mvnrnd(theta(i, :), lambda_seq(i) * Sigma(:, :, i));
    log_alpha = min(0, log_pi(theta_prop) - log_pi(theta(i, :)));

    % Accept or reject
    if log(rand()) < log_alpha
        theta(i+1, :) = theta_prop;
        nacc = nacc + 1;
    else
        theta(i+1, :) = theta(i, :);
    end

    % Update parameters
    lambda_seq(i+1) = exp(log(lambda_seq(i)) + gamma_sequence(i) * (exp(log_alpha) - alpha_star));
    mu(i+1, :) = mu(i, :) + gamma_sequence(i) * (theta(i+1, :) - mu(i, :));
    Sigma(:, :, i+1) = Sigma(:, :, i) + gamma_sequence(i) * ((theta(i+1, :)' - mu(i, :)') * (theta(i+1, :) - mu(i, :)) - Sigma(:, :, i));
end

store = theta(2:end, :);
acc = nacc / nits;
end

function [res] = log_pi(x)
res = mixture_gaussian_target(x');
end
