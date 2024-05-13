function policy_plot(policy, title_str, lb, ub)

if nargin < 2
    title_str = "Policy Plot";
end
if nargin < 3
    lb = -6;
end
if nargin < 4
    ub = 6;
end

X = lb:.1:ub;
[~, len] = size(X);
grid_sample = [X; X]';

for i = 1:len
    actions(i,:) = getAction(policy,grid_sample(i,:));
end

plot(X, actions);
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 17)
title(title_str, 'Interpreter', 'latex', 'FontSize', 17);

end
