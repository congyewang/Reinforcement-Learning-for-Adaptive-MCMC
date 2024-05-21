function res = discrete_target(x)

if x >= -6 && x <= -4
    res = log(0.25);
elseif x >= 4 && x <= 6
    res = log(0.25);
else
    res = -1e10;
end

end
