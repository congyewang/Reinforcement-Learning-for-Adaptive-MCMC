clearvars;
clc;
rng(0);

%% Add Packages
addpath(genpath('../../src/rlmcmc/'));

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
actor = make_actor_identity(env, pretrain_sample);

%% Set TD3
agent = rlTD3Agent(actor,critic);

agent.AgentOptions.ExplorationModel.StandardDeviation = zeros(bitshift(sample_dim, 1), 1);
agent.AgentOptions.ExperienceBufferLength = 10^6;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 10;
agent.AgentOptions.ResetExperienceBufferBeforeTraining = true;

%% Training
trainOpts = rlTrainingOptions( ...
    "MaxEpisodes", 140, ...
    "StopTrainingCriteria", "EpisodeCount", ...
    "SaveAgentCriteria", "EpisodeCount", ...
    "SaveAgentValue", 1, ...
    "SaveAgentDirectory", "savedAgents", ...
    "Plots", "none" ...
    );
trainingInfo = train(agent,env,trainOpts);

%% Save Store
save_store(env, 'train');
