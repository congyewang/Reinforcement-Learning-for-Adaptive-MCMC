function out = my_normpdf(x,m,s)

out = (2*pi*(s^2))^(-1/2) * exp( - (x-m)^2 / (2*(s^2)));

end