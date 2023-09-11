% true J function, computed using cubature wrt P
function out = J(t)

npt = 20; % number of cubature nodes
out = (2*pi)^(-1/2) * GaussHermite(@(x) Q(x,t),npt); % cubature

end