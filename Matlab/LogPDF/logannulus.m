function res = logannulus(x)
R=10;
sigma=2;
r = sqrt(sum(square(x)));
res = log(1/(2*pi*sigma^2)) - 0.5*(r-R)^2/sigma^2 + log(r<=R);
end