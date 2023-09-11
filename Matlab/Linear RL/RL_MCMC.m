function [x,t,w] = RL_MCMC()

x0 = 10; % initial state
t0 = 0.01; % initial policy parameter
w0 = 0.01; % initial Q parameter
g = 0; % discount factor
a = @(n) 0.01; % t learning rate
b = @(n) 0.01; % w learning rate

n = 100000; % number of iterations
x = zeros(n+1,1); % MCMC states
t = zeros(n+1,1); % policy parameters
w = zeros(n+1,1); % Q parameters
x(1) = x0;
t(1) = t0;
w(1,:) =  w0;

[Q,Q_dw,~] = Qfn(w(1,:),t(1)); % initialise Q
[s,s_dt] = sigma(t(1)); % initialise policy

for i = 1:n
    
    % epsilon greedy (not necessary if parametric Q_w is very good)
    e = 0.9 + 0.2 * rand();
    s_pert = @(x) e * s(x);
    [x(i+1),x_prop,alpha] = MH(x(i),s_pert);
    om = omega(x(i),x_prop,s,s_pert);
    
    % policy evaluation
    Qi = om * ( (x(i)-x_prop)^2 * alpha + g * Q(x(i+1),s(x(i+1))) );
    w(i+1,:) = w(i,:) + a(i) * (Qi - Q(x(i),s_pert(x(i)))) * Q_dw(x(i),s_pert(x(i)))';
    [Q,Q_dw,Q_da] = Qfn(w(i+1,:),t(i));
    
    % policy improvement
    t(i+1) = t(i) + b(i) * om * s_dt(x(i))' * Q_da(x(i),s(x(i)));
    [s,s_dt] = sigma(t(i+1));
    
end

end

% policy - the standard deviation of the MH proposal
function [s,s_dt] = sigma(t)
    
s = @(x) t; % policy
s_dt = @(x) 1; % gradient of policy wrt its parameters

end

% parametric approximation to the Q function
function [Q,Q_dw,Q_da] = Qfn(w,t)

%[s,s_dt] = sigma(t);
%Q = @(x,a) (a - s(x)) * w(:,1) * s_dt(x) + w(:,2);
%Q_dw = @(x,a) [(a - s(x)) * s_dt(x); ...
%               1];
%Q_da = @(x,a) w(:,1) * s_dt(x);

Q = @(x,a) w(:,1) .* a.^2 ./ (1 + (a-abs(x)).^2); % Q function
Q_dw = @(x,a) a.^2 ./ (1 + (a-abs(x)).^2); % (d/dw) Q
Q_da = @(x,a) ((1 + (a-abs(x)).^2) .* w(:,1) .* (2*a) ...
               - a.^2 .* 2 .* (a-abs(x))) ...
               / ((1 + (a-abs(x)).^2).^2); % (d/da) Q

end

% Metropolis--Hastings update
function [x_new,x_prop,alpha] = MH(x_old,s)
    
% propose a new state
x_prop = x_old + s(x_old) * randn();

% accept/reject
log_ratio = log_normpdf(x_prop) ...
            + log_normpdf((x_prop-x_old)/s(x_prop)) ...
            - log_normpdf(x_old) ...
            - log_normpdf((x_old-x_prop)/s(x_old));
if log(rand()) < log_ratio
    x_new = x_prop;
else 
    x_new = x_old;
end
alpha = min(1,exp(log_ratio));
    
end

% importance weights for epsilon-greedy
function out = omega(x_old,x_prop,s,s_pert)
    
% probability under s    
log_ratio = log_normpdf(x_prop) ...
            + log_normpdf((x_prop-x_old)/s(x_prop)) ...
            - log_normpdf(x_old) ...
            - log_normpdf((x_old-x_prop)/s(x_old));
alpha = min(1,exp(log_ratio));
prob_s = log_normpdf((x_old-x_prop)/s(x_old)) * alpha;

% probability under s_pert    
log_ratio = log_normpdf(x_prop) ...
            + log_normpdf((x_prop-x_old)/s_pert(x_prop)) ...
            - log_normpdf(x_old) ...
            - log_normpdf((x_old-x_prop)/s_pert(x_old));
alpha = min(1,exp(log_ratio));
prob_s_pert = log_normpdf((x_old-x_prop)/s_pert(x_old)) * alpha;

% importance weight
out = prob_s / prob_s_pert;

end

% log of normal pdf
function out = log_normpdf(x)

    out = -(1/2) * log(2*pi) - (1/2) * x^2;

end