function [x, mu, Sigma, log_lambda] = AdaptiveMetropolis(logpdf,d,n,rate)
%{
    Adaptive Metropolis Algorithm.
    logpdf - function handle to the pdf, taking column vector input.
    d - dimension of the RV to be sampled.
    n - the number of samples desired.
    rate - the exponent in the learning rate.
%}

if nargin < 3
    n = 1000;
end
if nargin < 4
    rate = 0.5;
end

%% initialisation
x = zeros(d,n);
mu = zeros(d,n);
Sigma = zeros(d,d,n); Sigma(:,:,1) = eye(d);
log_lambda = zeros(n,1);

%% learning rate
gamma = @(i) 0.5 * i^(-rate);

%% target acceptance rate
alpha_goal = 0.234;

%% iterations
accept_status = false(n,1);
for i = 2:n

    %% MH step
    proposal_cov = exp(log_lambda(i-1)) * Sigma(:,:,i-1);
    proposal_cov = nearestSPD(proposal_cov);
    x_star = mvnrnd( x(:,i-1)', proposal_cov )';
    log_alpha = min( 0 , logpdf(x_star) - logpdf(x(:,i-1)) );
    if log(rand()) < log_alpha
        x(:,i) = x_star;
        accept_status(i) = true;
    else
        x(:,i) = x(:,i-1);
    end

    %% adaptation
    %log_lambda(i) = log( 2.38^2 / d );
    log_lambda(i) = log_lambda(i-1) ...
        + gamma(i) * (exp(log_alpha) - alpha_goal);
    mu(:,i) = mu(:,i-1) + gamma(i) * (x(:,i) - mu(:,i-1));
    Sigma(:,:,i) = Sigma(:,:,i-1) ...
        + gamma(i) * ( (x(:,i) - mu(:,i-1)) * ((x(:,i) - mu(:,i-1))') ...
        - Sigma(:,:,i-1) );

end

end
