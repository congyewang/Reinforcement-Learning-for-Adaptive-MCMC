function policy_plot_sample_2D(policy, log_target_pdf, color_list, nits, lb, ub, sample_path)

if nargin < 4
    color_list = [1,0,0,0.1];
end
if nargin < 5
    nits = 10;
end

if nargin < 6
    lb = -10;
end
if nargin < 7
    ub = 10;
end
if nargin < 8
    sample_path = "Data/train_store_accepted_sample.mat";
end

%% Extract Data
sample_data = load(sample_path).data;
[sample_data_unique, ~, ~] = unique(sample_data, 'rows', 'stable');
half_index = ceil(size(sample_data_unique, 1) / 2);
sample_data_unique_half = sample_data_unique(half_index:end, :);
random_order = randperm(size(sample_data_unique_half, 1));
sample_data_unique_half_shuffled = sample_data_unique_half(random_order, :);

%% Initial Store
obs = [sample_data_unique_half_shuffled(end-nits+1:end,:),sample_data_unique_half_shuffled(end-nits+1:end,:)];
policy_res = zeros(nits, 2);

%% Plot Mean
for i = 1:nits
    res = getAction(policy, obs(i,:));
    policy_res(i, 1:2) = res(1:2)';

    plot_samples_3D(obs(i,1:2), policy_res(i, 1:2), log_target_pdf, color_list, lb, ub);
end

end
