function policy_save(policy, data_file_name, lb, ub)

if nargin < 2
    lb = -6;
end
if nargin < 3
    ub = 6;
end

X = linspace(lb, ub, 201);
[~, len] = size(X);
grid_sample = [X; X]';

for i = 1:len
    actions(i,:) = getAction(policy,grid_sample(i,:));
end

save(sprintf("Data/Policy/%s.mat", data_file_name), 'actions');

end
