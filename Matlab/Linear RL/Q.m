function out = Q(x,a,sigma,rew_length,gamma)

rng(0);

N = 100; % Monte Carlo approximation
out = 0;
X(1) = x;
for n = 1:N
    X(2) = MH(x,@(x) a); 
    X(2:rew_length) = MCMC(X(2),sigma,rew_length-1);
    out = out + (1/N) * cum_r(X,gamma);
end

end


   