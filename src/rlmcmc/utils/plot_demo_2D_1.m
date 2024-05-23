function plot_demo_2D_1(xlims, n_break1, n_break2)

if nargin < 1
    xlims = [-10,10];
end
if nargin < 2
    n_break1 = 9;
end
if nargin < 3
    n_break2 = 50;
end

close all;

%% Path Config

avg_reward_path = "Data/data_rl_average_episode_reward_moving_window.mat";
target_path = "Data/target.mat";
mala_path = "Data/mala.mat";
nuts_path = "Data/nuts.mat";
sample_path = "Data/train_store_accepted_sample.mat";

%% 3D target
plot_log_pdf = @(x) 0.5 * mvnpdf(x, [-5, 0], eye(2)) + 0.5 * mvnpdf(x, [5, 0], eye(2));

%% Plot
figure()

ylims = xlims;

% Average
avg_reward = load(avg_reward_path).data_rl_average_episode_reward_moving_window;
subplot(2,4,1)
n_vals = 500 * (0:139);
format1 = 'k-';
format2 = 'r-';
format3 = 'b-';
plot(n_vals(1:n_break1),avg_reward(1:n_break1),format1); hold on;
plot(n_vals(n_break1:n_break2),avg_reward(n_break1:n_break2),format2)
plot(n_vals(n_break2:end),avg_reward(n_break2:end),format3)
xlabel('$n$')
ylabel('Reward $r_n$')

% Policy 1
subplot(2,4,5)
load_agent1 = load(['savedAgents/Agent',num2str(1,'%u'),'.mat']);
generatePolicyFunction(load_agent1.saved_agent,"MATFileName",'load_agentData1.mat');
policy1 = coder.loadRLPolicy("load_agentData1.mat");
policy_plot_2D(policy1, "Step 1 Policy", plot_log_pdf, [0,0,0,0.01]);

ax = gca;
ax.ZTick = [];

xlabel('$x_{1}$')
ylabel('$x_{2}$')
zlabel('$\phi(x)$')
xlim(xlims)
ylim(ylims)

% Policy 3
subplot(2,4,2)
load_agent3 = load(['savedAgents/Agent',num2str(3,'%u'),'.mat']);
generatePolicyFunction(load_agent3.saved_agent,"MATFileName",'load_agentData3.mat');
policy3 = coder.loadRLPolicy("load_agentData3.mat");
policy_plot_2D(policy3, "Step 1500 Policy", plot_log_pdf, [1,0,0,0.1]);

ax = gca;
ax.ZTick = [];

set(ax, 'LooseInset', get(ax, 'TightInset'));

xticks([])
yticks([])
zlabel('$\phi(x)$')
xlim(xlims)
ylim(ylims)

% Policy 140
subplot(2,4,6)

load_agent140 = load(['savedAgents/Agent',num2str(140,'%u'),'.mat']);
generatePolicyFunction(load_agent140.saved_agent,"MATFileName",'load_agentData140.mat');
policy140 = coder.loadRLPolicy("load_agentData140.mat");
policy_plot_2D(policy140, "Step 70000 Policy", plot_log_pdf, [0,0,1,0.1]);

ax = gca;
ax.ZTick = [];

set(ax, 'LooseInset', get(ax, 'TightInset'));

xlabel('$x_{1}$')
ylabel('$x_{2}$')
zlabel('$\phi(x)$')
xlim(xlims)
ylim(ylims)

%Target
target_data = load(target_path).target_data;
subplot(2,4,3)
x = linspace(xlims(1), xlims(2), 100);
y = linspace(xlims(1), xlims(2), 100);
[X, Y] = meshgrid(x, y);
contour(X, Y, target_data)
colormap(gray)
xlabel('$p$')
xlim(xlims)
xticks([])
yticks([])
grid off;

% Mala
mala_data = load(mala_path).mala_data;
subplot(2,4,4)
yyaxis right
plot( ...
    mala_data(end-5000:end, 1), ...
    mala_data(end-5000:end, 2), ...
    'Marker', 'o', ...
    'LineStyle', '-', ...
    'Color', [0, 0, 0, 0.1] ...
);
xlabel('MALA')
xlim(xlims)
xticks([])
set(gca, 'YColor', 'k');
yyaxis left
set(gca, 'YColor', 'k');
yticks([]);
grid off;

% NUTS
nuts_data = load(nuts_path).nuts_data;
subplot(2,4,7)
plot( ...
    nuts_data(end-5000:end, 1), ...
    nuts_data(end-5000:end, 2), ...
    'Marker', 'o', ...
    'LineStyle', '-', ...
    'Color', [0, 0, 0, 0.1] ...
);
xlabel('NUTS')
xlim(xlims)
yticks([])
grid off;

% RL
data = load(sample_path).data;
subplot(2,4,8)
yyaxis right
plot( ...
    data(end-5000:end, 1), ...
    data(end-5000:end, 2), ...
    'Marker', 'o', ...
    'LineStyle', '-', ...
    'Color', [0, 0, 0, 0.1] ...
);
xlabel('RLMH')
xlim(xlims)
set(gca, 'YColor', 'k');
yyaxis left
set(gca, 'YColor', 'k');
yticks([]);
grid off;

%% Save to PDF
printpdf_2D('demo_plot_1')

end