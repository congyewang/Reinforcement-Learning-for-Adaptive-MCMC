function plot_demo_nn(xlims, n_break1, n_break2, polyline)

if nargin < 1
    xlims = [-10,10];
end
if nargin < 2
    n_break1 = 20;
end
if nargin < 3
    n_break2 = 50;
end
if nargin < 4
    polyline = true;
end

%% Path Config

avg_reward_path = "Data/data_rl_average_episode_reward_moving_window.mat";
policy_path = "Data/Policy";
sample_path = "Data/train_store_accepted_sample.mat";

%% Plot
figure()

ylims = xlims;
x_vals = linspace(xlims(1),xlims(2),201);

avg_reward = load(avg_reward_path).data_rl_average_episode_reward_moving_window;
subplot(2, 3, [1, 4])
n_vals = 500 * (0:139);
format1 = 'k-';
format2 = 'r-';
format3 = 'b-';
plot(n_vals(1:n_break1),avg_reward(1:n_break1),format1); hold on;
plot(n_vals(n_break1:n_break2),avg_reward(n_break1:n_break2),format2)
plot(n_vals(n_break2:end),avg_reward(n_break2:end),format3)
xlabel('$n$')
ylabel('Reward $r_n$')

initial_policy = load(sprintf('%s/%s', policy_path, 'policy1.mat')).actions;
subplot(3, 3, 2)
plot(x_vals,initial_policy(:,1),format1)
xticks([])
ylabel('$\phi(x)$')
xlim(xlims)
ylim(ylims)
if polyline
    xline(-5,'k:')
    xline(5,'k:')
    yline(-5,'k:')
    yline(5,'k:')
end

subplot(3, 3, 5)
for i = n_break1:n_break2
    median_policy = load(sprintf('%s/policy%d.mat', policy_path, i)).actions;
    patchline(x_vals,median_policy(:,1),'edgecolor','r','edgealpha',0.2); hold on
end
xticks([])
ylabel('$\phi(x)$')
xlim(xlims)
ylim(ylims)
if polyline
    xline(-5,'k:')
    xline(5,'k:')
    yline(-5,'k:')
    yline(5,'k:')
end

subplot(3, 3, 8)
for i = n_break2+1:140
    final_policy = load(sprintf('%s/policy%d.mat', policy_path, i)).actions;
    patchline(x_vals,final_policy(:,1),'edgecolor','b','edgealpha',0.2); hold on
end
xlabel('$x$')
ylabel('$\phi(x)$')
xlim(xlims)
ylim(ylims)

if polyline
    xline(-5,'k:')
    xline(5,'k:')
    yline(-5,'k:')
    yline(5,'k:')
end

data = load(sample_path).data;
subplot(2, 3, [3,6])
histogram(data(end-5000:end),'BinEdges',x_vals,'DisplayStyle','stairs','FaceColor',"none",'EdgeColor','k','EdgeAlpha',1)
xlabel('RLMH')
xlim(xlims)
yticks([])
if polyline
    xline(-5,'k:')
    xline(5,'k:')
end

%% Save to PDF
printpdf('demo_plot')

end