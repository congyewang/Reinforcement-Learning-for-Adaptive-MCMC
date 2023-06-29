function [store, acc] = policy_mh(theta_start, policy_cov, log_pi, nits)
d = numel(theta_start);
store = zeros(nits+1, d);
nacc = 0;
theta_curr = theta_start;
log_pi_curr = log_pi(theta_curr);
store(1,:) = theta_curr;

for i = 1:nits
    % Current State
    sigma2_curr = policy_cov(theta_curr);

    % Proposed State
    theta_prop = mvnrnd(theta_curr, sigma2_curr);
    sigma2_prop = policy_cov(theta_prop);
    log_pi_prop = log_pi(theta_prop);

    log_alpha = log_pi_prop - log_pi_curr + logmvnpdf(theta_curr, theta_prop, sigma2_prop) - logmvnpdf(theta_prop, theta_curr, sigma2_curr);

    if log(rand()) < log_alpha
        theta_curr = theta_prop;
        log_pi_curr = log_pi_prop;
        nacc = nacc + 1;
    end
    store(i+1,:) = theta_curr;

end
acc = nacc/nits;

return
end