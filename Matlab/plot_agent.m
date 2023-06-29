load('Agent50.mat');
agent = saved_agent;

%% Plot Contour Policy Function
s0 = -10:.1:10;
trace_sigma2 = zeros(numel(s0),numel(s0));
for i = 1:numel(s0)
    for j = 1:numel(s0)
        formatted_state = {[s0(i);s0(j)]};
        actions = cell2mat(getAction(agent, formatted_state));
        a = actions(1);
        c = actions(2);
        sigma2 = [a, 0; 0, c] * [a, 0; 0, c];
        trace_sigma2(i,j) = trace(sigma2);
    end
end

figure;
contour(s0, s0, log(trace_sigma2), 100);
xlabel('x1');
ylabel('x2');
zlabel('tr($\sigma^2$)','interpreter','latex');
title('Policy Plot');

%% Plot Quiver Policy Function
s0 = linspace(-2, 2, 100);
s1 = linspace(-5, 5, 100);
X = zeros(numel(s1), numel(s0));
Y = zeros(numel(s1), numel(s0));
U = zeros(numel(s1), numel(s0));
V = zeros(numel(s1), numel(s0));
trace_sigma2 = zeros(numel(s0),numel(s1));
for i = 1:numel(s0)
    for j = 1:numel(s1)
        formatted_state = {[s0(i);s1(j)]};
        actions = cell2mat(getAction(agent, formatted_state));
        a = actions(1);
        % b = actions(2);
        c = actions(2);
        % sigma2 = [a, b; 0, c] * [a, 0; b, c];
        sigma2 = [a, 0; 0, c] * [a, 0; 0, c];
trace_sigma2(i,j) = trace(sigma2);
        X(i,j) = s0(i);
        Y(i,j) = s1(j);
        [v,lam] = eigs(sigma2,1);
        U(i,j) = lam * v(1);
        V(i,j) = lam * v(2);
    end
end

figure;
q = quiver(X,Y,U,V);
q.ShowArrowHead = 'off';
q.Marker = 'none';

xlabel('x1');
ylabel('x2');
zlabel('tr($\sigma^2$)','interpreter','latex');
title('Policy Quiver Plot');
