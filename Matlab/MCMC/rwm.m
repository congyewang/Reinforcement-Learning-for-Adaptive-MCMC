function [store, acc] = rwm(sigma2_curr, sigma2_prop, theta_start, log_pi, nits)
d = length(theta_start);
store = zeros(nits+1, d);
log_pi_theta_prop = zeros(nits, d);
nacc = 0;
theta_curr = theta_start;
log_pi_curr = log_pi(theta_curr);
store(1,:) = theta_curr;
for i = 1:nits
    theta_prop = mvnrnd(theta_curr, sigma2_curr );
    log_pi_theta_prop(i,:) = theta_prop;
    log_pi_prop = log_pi(theta_prop);
    log_alpha = log_pi_prop - log_pi_curr + logmvnpdf(theta_curr, theta_prop, sigma2_prop) - logmvnpdf(theta_prop, theta_curr, sigma2_curr);
    if (log(rand())<log_alpha)
        theta_curr = theta_prop;
        log_pi_curr = log_pi_prop;
        nacc = nacc + 1;
    end
    store(i+1,:) = theta_curr;
end
acc = nacc/nits;
store = store(2:end, :);
end
