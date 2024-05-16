function logpdf = skew_normal_target(x, alpha, mu, var)

if nargin < 2
    alpha = 2;
end
if nargin < 3
    mu = 0;
end
if nargin < 4
    var = 1;
end

if var <= 0
    error('var must be positive.');
end

std = sqrt(var);

z = (x - mu) / std;

log_phi = -0.5 * log(2 * pi) - 0.5 * z.^2;
Phi_alpha_z = 0.5 * (1 + erf(alpha * z / sqrt(2)));

Phi_alpha_z(Phi_alpha_z == 0) = realmin;

logpdf = log(2) - log(std) + log_phi + log(Phi_alpha_z);

end
