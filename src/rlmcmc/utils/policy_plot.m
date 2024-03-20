function policy_plot(policy)

X = -5:.1:5;
[~, len] = size(X);
grid_sample = [X; X]';

for i = 1:len
    actions(i,:) = getAction(policy,grid_sample(i,:));
end

plot(X, actions);
xlabel('x')
title('Policy Plot');

end