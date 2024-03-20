function res = gaussian_target(x, sample_dim)

if nargin < 2
    sample_dim = 2;
end

res = logmvnpdf(x', zeros(1, sample_dim), eye(sample_dim));

end
