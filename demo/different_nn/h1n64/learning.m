clearvars;
clc;
rng(0);

%% Add Packages
addpath(genpath('../../../src/rlmcmc/'));

%% Add Log Target PDF
log_target_pdf = @(x) mixture_gaussian_target(x, [0.5, 0.5], [-5; 5], cat(3, eye(1), eye(1)));

%% get (approx) mean (mu) and covariance (Sigma) from adaptive mcmc
pretrain_nits = 100;
pretrain_sample = normrnd(0, 25, [pretrain_nits, 1]); % data for pre-training

%% Set environment - use (approx) mean (mu) and covariance (Sigma) to inform proposal
sample_dim = 1;
env = RLMHEnvDemo(log_target_pdf, zeros(sample_dim, 1), zeros(sample_dim, 1), eye(1));

%% Set Critic
critic = make_critic(env);

%% Set Actor
actor = make_actor_identity(env, pretrain_sample, 64);

%% Set DDPG
agent = rlDDPGAgent(actor,critic);

agent.AgentOptions.NoiseOptions.StandardDeviation = zeros(bitshift(sample_dim, 1), 1);
agent.AgentOptions.ExperienceBufferLength=10^6;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1e-4;
agent.AgentOptions.ResetExperienceBufferBeforeTraining = true;

%% Training
trainOpts = rlTrainingOptions( ...
    "MaxEpisodes", 140, ...
    "StopTrainingCriteria", "EpisodeCount", ...
    "SaveAgentCriteria", "EpisodeCount", ...
    "SaveAgentValue", 1, ...
    "SaveAgentDirectory", "savedAgents" ...
    );
trainingInfo = train(agent,env,trainOpts);

%% Save Store
save_store(env, 'train');

%% Plot Policy
fig1 = figure;
load_agent1 = load(['savedAgents/Agent',num2str(1,'%u'),'.mat']);
generatePolicyFunction(load_agent1.saved_agent,"MATFileName",'load_agentData1.mat');
policy1 = coder.loadRLPolicy("load_agentData1.mat");
policy_plot(policy1, "Step 1 Policy");
axis square;
exportgraphics(fig1, 'Policy_Ep1.pdf');

fig3 = figure;
load_agent3 = load(['savedAgents/Agent',num2str(3,'%u'),'.mat']);
generatePolicyFunction(load_agent3.saved_agent,"MATFileName",'load_agentData3.mat');
policy3 = coder.loadRLPolicy("load_agentData3.mat");
policy_plot(policy3, "Step 1500 Policy");
axis square;
exportgraphics(fig3, 'Policy_Ep3.pdf');

fig140 = figure;
load_agent140 = load(['savedAgents/Agent',num2str(140,'%u'),'.mat']);
generatePolicyFunction(load_agent140.saved_agent,"MATFileName",'load_agentData140.mat');
policy140 = coder.loadRLPolicy("load_agentData140.mat");
policy_plot(policy140, "Step 70000 Policy");
axis square;
exportgraphics(fig140, 'Policy_Ep140.pdf');
