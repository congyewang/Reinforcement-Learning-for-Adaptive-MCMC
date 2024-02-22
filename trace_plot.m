function [] = trace_plot(env)
figure;

[~, nCols] = size(env.StoreState);
Sample = zeros(nCols, 2);
for i = 1:nCols
    Sample(i, :) = env.StoreState{i};
end

acc = sum(cell2mat(env.StoreAcceptedStatus)) / length(cell2mat(env.StoreAcceptedStatus));

plot(Sample(:, 1));
titleText = sprintf('Trace Plot (Acceptance Rate=%.4f)', acc);
title(titleText);

end