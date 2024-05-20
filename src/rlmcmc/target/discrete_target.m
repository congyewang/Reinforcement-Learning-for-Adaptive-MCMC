function res = discrete_target(x)

if x >= -5 && x <= -4
    res = log(0.5);
elseif x >= 4 && x <= 5
    res = log(0.5);
else
    res = -1e10;
end

end
