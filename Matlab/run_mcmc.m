clear;
clc;


theta_start = [0.0, 0.0];
log_pi = @(x) log(mvnpdf(x, [4.0,5.0], eye(2,2)));
policy_cov = @(x) optimal_policy_cov(x);
nits = 5000;

[store, acc] = policy_mh(theta_start, policy_cov, log_pi, nits);

figure;
plot(store(:,1), store(:,2));
distances = jump_distance(store);
disp(mean(distances));

grad_log_pi = @(x) [4.0, 5.0] - x;
alpha = repmat(0.3, [1, 10]);
epoch = [repmat(1000, [1, 9]), nits];
[x_mala_adapt, ~, ~, a_mala_adapt, ~, ~] = mala_adapt(log_pi, grad_log_pi, theta_start, 0.1, eye(2), alpha, epoch);

figure;
plot(x_mala_adapt{end}(:,1), x_mala_adapt{end}(:,2));
distances_mala_adapt = jump_distance(x_mala_adapt{end});
disp(mean(distances_mala_adapt));
