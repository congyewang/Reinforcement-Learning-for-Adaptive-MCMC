function [] = policy_plot()
figure;

current_sample = -3:.1:3;
[~, len] = size(current_sample);
proposed_sample = zeros(1, len);
grid_sample = [current_sample; proposed_sample]';

store_action = zeros(len, 2);

for i = 1:len
    store_action(i,:) = evaluatePolicy(grid_sample(i,:));
end

plot(current_sample, store_action);
title('Policy Plot');

end