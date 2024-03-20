function policy_plot_2D(policy)

x_n_1 = [-5, -3, 0, 3, 5];
x_n_2 = [-5, -3, 0, 3, 5];

[X_N_1, X_N_2] = ndgrid(x_n_1, x_n_2);

X_N_1_col = X_N_1(:);
X_N_2_col = X_N_2(:);

x_n = [X_N_1_col, X_N_2_col];

obs = [x_n, zeros(size(x_n))];
nits = size(obs, 1);
policy_res = zeros(nits, 5);

for i = 1:nits
    res = getAction(policy, obs(i,:));
    policy_res(i, 1:2) = obs(i,1:2);
    policy_res(i, 3:4) = obs(i,1:2) + res(1:2)';
    policy_res(i, 5) = res(end);

    create_cov_ellipse(policy_res(i, 5)^2 * eye(2), policy_res(i, 3:4), obs(i,1:2));
end

end