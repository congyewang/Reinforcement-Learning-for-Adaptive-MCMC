function out = cum_r(X,gamma)
    out = 0;
    for i = 1:length(X)-1
        out = out + gamma^(i-1) * (X(i+1)-X(i))^2;
    end
end