function [] = action_plot(env)
figure;

[~, nCols] = size(env.StoreAction);
Action = zeros(nCols, 2);
for i = 1:nCols
    Action(i, :) = env.StoreAction{i};
end

plot(Action);
title('Action Plot');

end