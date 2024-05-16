function res = discrete_target(x)

if x >= -4 && x <= -3
    res = log(0.25);
elseif x >= -1 && x <= 1
    res = log(0.5);
elseif x >= 3 && x <= 4
    res = log(0.25);
else
    res = -realmax;
end

end
