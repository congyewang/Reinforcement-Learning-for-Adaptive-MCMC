function [store, acc] = adaptive_mh(nits, theta_start, log_pi, agent)
d = length(theta_start);

store = zeros(nits+1, d);
log_pi_theta_prop = zeros(nits, d);

nacc = 0;

theta_curr = theta_start;
theta_prop = theta_start;

log_pi_curr = log_pi(theta_curr);
store(1,:) = theta_curr;

for i = 1:nits
    formatted_theta_prop = {[theta_prop(1);theta_prop(2)]};
    sigma2_prop = diag(cell2mat(getAction(agent, formatted_theta_prop))).^2;
    theta_prop = mvnrnd(theta_curr, sigma2_prop);

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
