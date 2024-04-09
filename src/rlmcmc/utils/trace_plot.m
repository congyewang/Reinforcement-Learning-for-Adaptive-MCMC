function [] = trace_plot(env)
figure;

[~, n_cols] = size(env.store_accepted_sample);
sample_mat = zeros(n_cols, env.sample_dim, 1);

for i = 1:n_cols
    sample_mat(i, :) = env.store_accepted_sample{i};
end

acc = sum(cell2mat(env.store_accepted_status)) / length(cell2mat(env.store_accepted_status));

if env.sample_dim == 1
    plot( ...
        sample_mat(:, 1), ...
        'Marker', 'o', ...
        'LineStyle', '-', ...
        'Color', [0, 0.447, 0.741, 0.1] ...
        );
elseif env.sample_dim == 2
    plot( ...
        sample_mat(:, 1), ...
        sample_mat(:, 2), ...
        'Marker', 'o', ...
        'LineStyle', '-', ...
        'Color', [0, 0.447, 0.741, 0.1] ...
        );
else
    hold on
    for j = 1:env.sample_dim
        plot(sample_mat(:, j));
    end
    hold off
end

titleText = sprintf('Trace Plot (Acceptance Rate=%.4f)', acc);
title(titleText);

end