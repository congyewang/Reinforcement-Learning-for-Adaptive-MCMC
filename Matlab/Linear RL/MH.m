function x_new = MH(x_current,sigma)

% propose a new state
x_proposed = x_current + sigma(x_current) * randn();

% accept/reject
log_alpha = log_normpdf(x_proposed) ...
            + log_normpdf((x_proposed-x_current)/sigma(x_proposed)) ...
            - log_normpdf(x_current) ...
            - log_normpdf((x_current-x_proposed)/sigma(x_current));
if log(rand()) < log_alpha
    x_new = x_proposed;
else 
    x_new = x_current;
end
    
end


function out = log_normpdf(x)

    out = -(1/2) * log(2*pi) - (1/2) * x^2;

end