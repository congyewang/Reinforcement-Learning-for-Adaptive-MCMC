load('Agent50.mat');
agent = saved_agent;

%% Plot Contour Policy Function
s0 = -10:.1:10;
trace_sigma2 = zeros(numel(s0),numel(s0));
a_store = zeros(numel(s0),numel(s0));
c_store = zeros(numel(s0),numel(s0));
for i = 1:numel(s0)
    for j = 1:numel(s0)
        formatted_state = {[s0(i);s0(j)]};
        actions = cell2mat(getAction(agent, formatted_state));
        a = actions(1);
        c = actions(2);
        sigma2 = [a, 0; 0, c] * [a, 0; 0, c];
        trace_sigma2(i,j) = trace(sigma2);
        a_store(i,j) = a^2;
        c_store(i,j) = c^2;
    end
end

% figure;
% contour(s0, s0, log(trace_sigma2), 100);
% xlabel('x1');
% ylabel('x2');
% zlabel('tr($\sigma^2$)','interpreter','latex');
% title('Policy Plot');

figure;
plot(s0,a_store(1,:));
xlabel('x1');
ylabel('a');
zlabel('a$)','interpreter','latex');
title('Policy Plot');

figure;
plot(s0,c_store(1,:));
xlabel('x1');
ylabel('c');
zlabel('a$)','interpreter','latex');
title('Policy Plot');