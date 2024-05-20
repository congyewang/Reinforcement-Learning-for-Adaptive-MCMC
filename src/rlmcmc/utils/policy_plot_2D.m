function policy_plot_2D(policy, title_str, log_target_pdf, x_n_1, x_n_2, lb, ub)
if nargin < 4
    x_n_1 = [-7, -5, -3, 0, 3, 5, 7];
end

if nargin < 5
    x_n_2 = [-1, 0, 1];
end

if nargin < 6
    lb = -6;
end
if nargin < 7
    ub = 6;
end

[X_N_1, X_N_2] = ndgrid(x_n_1, x_n_2);

X_N_1_col = X_N_1(:);
X_N_2_col = X_N_2(:);

x_n = [X_N_1_col, X_N_2_col];

obs = [x_n, zeros(size(x_n))];
nits = size(obs, 1);
policy_res = zeros(nits, 2);

for i = 1:nits
    res = getAction(policy, obs(i,:));
    policy_res(i, 1:2) = res(1:2)';

    plot_samples_3D(obs(i,1:2), policy_res(i, 1:2), log_target_pdf, lb, ub);
end

xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 17);
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 17);
title(title_str, 'Interpreter', 'latex', 'FontSize', 17);

end
