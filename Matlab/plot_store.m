% Plot MCMC Chain
figure;
plot(cell2mat(env.StoreState));
xlabel('x');
ylabel('t');
title('Trace Plot');

% Plot Distribution
figure;
histogram(cell2mat(env.StoreState), 100);
xlabel('x');
ylabel('t');
title('Histogram Plot');

% Plot Sigma
figure;
plot(cell2mat(env.StoreAction));
xlabel('$\sigma$','interpreter','latex');
ylabel('t');
title('Action Plot');

% Generate action using the trained actor
states = -5:0.1:5;
actions = zeros(size(states));
% StorePolicy = {};
% for i = -50:0.1:50
for i = 1:numel(states)
    formatted_state = {states(i)};
    actions(i) = cell2mat(getAction(agent, formatted_state));
end

figure;
plot(states, actions);
xlabel('States');
ylabel('Actions');
title('Policy Plot');
