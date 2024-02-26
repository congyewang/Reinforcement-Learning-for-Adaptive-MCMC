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

% Hist
figure;
if nCols < 10000
    bins_num = floor(1 + log(nCols) / log(2));
else
    bins_num = floor(2 * log(nCols)^(1/3));
end
histogram(Sample(:, 1), bins_num);

end