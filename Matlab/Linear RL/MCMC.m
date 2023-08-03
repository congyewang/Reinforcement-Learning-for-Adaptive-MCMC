function X = MCMC(x0,sigma,n)
    X = zeros(n,1);
    for i = 1:n
        if i == 1
            X(1) = x0;
        else
            X(i) = MH(X(i-1),sigma);
        end
    end
end