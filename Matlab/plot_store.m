plot(cell2mat(env.StoreState));

plot(cell2mat(env.StoreAction));

% Generate action using the trained actor
StorePolicy = {};
for i = -50:0.1:50
    StorePolicy{end+1} = cell2mat(getAction(actor, {i}));
end
plot(-50:0.1:50, cell2mat(StorePolicy));
