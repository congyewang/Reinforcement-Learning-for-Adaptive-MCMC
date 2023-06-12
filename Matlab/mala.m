function [x, accept_rate] = mala(logp, grad_logp, epsilon)
if(~exist('epsilon','var'))
    epsilon = 0.1;
end

% MALA
% Here I am following the formulation on Wikipedia (there was a sqrt(2)
% factor that I missed out on the whiteboard)
mu = @(x) x + epsilon * grad_logp(x); % proposal mean
var = @(x) 2 * epsilon; % proposal variance
logq = @(x,y) - 0.5 * log(2*pi*var(x)) - (y-mu(x))^2 / (2*var(x)); % log pdf of proposal (x -> y)
log_alpha = @(x,y) logp(y) - logp(x) + logq(y,x) - logq(x,y); % log acceptance ratio (x -> y)

N = 1000; % number of MCMC iterations
x = zeros(N+1,1);
x(1) = 3; % initial state for MCMC
accept_rate = 0; % keep track of the proportion of proposals that are accepted

for i = 1:N

    % propose a new state
    x_proposed = mu(x(i)) + randn(1) * sqrt(var(x(i)));

    % decide whether to accept or reject
    if log(rand()) < log_alpha(x(i),x_proposed)
        x(i+1) = x_proposed;
        accept_rate = accept_rate + (1/N);
    else
        x(i+1) = x(i);
    end
end

end