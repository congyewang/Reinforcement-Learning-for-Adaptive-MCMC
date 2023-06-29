function [grad_x, grad_y] = gradlogannulus(x)
R=10;
sigma=2;
r = sqrt(sum(square(x)));
grad_x = - (r - R) / sigma^2 * np.exp(-0.5*(r-R)^2/sigma^2) * (r<=R);
grad_y = np.zeros_like(grad_x);
end