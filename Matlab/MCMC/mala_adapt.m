function [x, g, p, a, h, c] = mala_adapt(fp, fg, x0, h0, c0, alpha, epoch)
n_ep = length(epoch);
x = cell(1, n_ep);
g = cell(1, n_ep);
p = cell(1, n_ep);
a = cell(1, n_ep);

h = h0;
c = c0;
[x{1}, g{1}, p{1}, a{1}] = mala(fp, fg, x0, h, c, epoch(1));

for i = 2:n_ep
    c = alpha(i - 1) * c + (1 - alpha(i - 1)) * cov(x{i - 1}(:, 1), x{i - 1}(:, 2));
    c = nearestSPD(c);

    ar = mean(a{i - 1});
    h = h * exp(ar - 0.57);

    x0_new = x{i - 1}(end, :);
    [x{i}, g{i}, p{i}, a{i}] = mala(fp, fg, x0_new, h, c, epoch(i));
end
end