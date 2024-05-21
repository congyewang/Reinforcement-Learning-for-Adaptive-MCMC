function policy_plot_2D(policy, log_target_pdf, color_list, x_n_1, x_n_2, lb, ub)
if nargin < 3
    color_list = [1,0,0,0.1];
end
if nargin < 4
    x_n_1 = [-7, -5, -3, 0, 3, 5, 7];
end

if nargin < 5
    x_n_2 = [-1, 0, 1];
end

if nargin < 6
    lb = -10;
end
if nargin < 7
    ub = 10;
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

    plot_samples_3D(obs(i,1:2), policy_res(i, 1:2), log_target_pdf, color_list, lb, ub);
end

end
