function [x, g, p, a] = mala(fp, fg, x0, h, c, n)
d = length(x0);
x = zeros(n, d);
g = zeros(n, d);
p = zeros(n, 1);
a = false(n, 1);
x(1, :) = x0;
g(1, :) = fg(x0);
p(1) = fp(x0);

hh = h ^ 2;
for i = 2:n
    mx = x(i - 1, :) + (hh / 2 * (c * g(i - 1, :)'))';
    s = hh * c;
    y = mvnrnd(mx, s);
    py = fp(y);
    gy = fg(y);
    my = y + (hh / 2 * (c * gy'))';
    qx = logmvnpdf(x(i - 1, :), my, s);
    qy = logmvnpdf(y, mx, s);
    acc_pr = (py + qx) - (p(i - 1) + qy);

    if acc_pr >= 0 || log(rand()) < acc_pr
        x(i, :) = y;
        g(i, :) = gy;
        p(i) = py;
        a(i) = true;
    else
        x(i, :) = x(i - 1, :);
        g(i, :) = g(i - 1, :);
        p(i) = p(i - 1);
    end
end
end
