% Load
% load('trained_agent.mat');

% Plot MCMC Chain
figure;
stored_states = cell2mat(env.StoreState);
x1 = stored_states(1, :);
x2 = stored_states(2, :);
plot(x1, x2);
xlabel('x1');
ylabel('x2');
title('Trace Plot');

% Plot MCMC Chain 3D
figure;
t = 1:length(x1);
plot3(t, x1, x2);

xlabel('t');
ylabel('x1');
zlabel('x2');
title('3D Trace Plot');

% Plot Distribution
figure;
histogram2(x1, x2, 'FaceColor', 'flat');
xlabel('x1');
ylabel('x2');
title('Histogram Plot');

% Plot Tr(Sigma2)
trace_sigma2 = zeros(1,length(env.StoreAction));

for i = 1:length(env.StoreAction)
    trace_sigma2(i) = trace(cell2mat(env.StoreAction(i)));
end

figure;
plot(trace_sigma2);
xlabel('t');
ylabel('tr($\sigma^2$)','interpreter','latex');
title('Trace of the Action Matrix');

% Generate action using the trained actor
states = -10:.1:10;
trace_sigma2 = zeros(1,numel(states));
for i = 1:numel(states)
    for j = 1:numel(states)
        formatted_state = {[states(i);states(j)]};
        actions = cell2mat(getAction(agent, formatted_state));
        a = actions(1);
        b = actions(2);
        c = actions(3);
        sigma2 = [a^2, 0; b, c^2] * [a^2, b; 0, c^2];
        trace_sigma2(i) = trace(sigma2);
    end
end

figure;
plot3(states, states, trace_sigma2);
xlabel('x1');
ylabel('x2');
zlabel('tr($\sigma^2$)','interpreter','latex');
title('Policy Plot');
