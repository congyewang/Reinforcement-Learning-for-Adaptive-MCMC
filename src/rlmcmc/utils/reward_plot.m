function [] = reward_plot(env)

reward_mat = cell2mat(env.store_reward);

figure;
plot(reward_mat);
title('Immediate Reward Plot');

figure;
plot(cumsum(reward_mat));
title('Cumulative Reward Plot');

end
