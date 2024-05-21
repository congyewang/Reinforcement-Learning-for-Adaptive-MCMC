clearvars;
clc;
rng(0);

%% Add Packages
addpath(genpath('..'));

%% Add Log Target PDF
log_target_pdf = @(x) disp('Please provide a log target pdf function.');

%% Get (Approx) Mean (mu) and Covariance (Sigma) from AMH
am_nits = 10000;
am_rate = {{ am_rate }};

[x_AMH,~,~,~,~] = AdaptiveMetropolis(log_target_pdf, sample_dim, am_nits, am_rate);

mu = mean(x_AMH(:,ceil(2*am_nits/3):end),2);
Sigma = cov(x_AMH(:,ceil(2*am_nits/3):end)');

actor_nits = am_nits - ceil(2*am_nits/3) + 1;
pretrain_sample = x_AMH(:,ceil(2*am_nits/3):end)'; % data for pre-training

%% Set environment - use (approx) mean (mu) and covariance (Sigma) to inform proposal
env = RLMHEnv(log_target_pdf, mu, mu, Sigma);

%% Set Critic
critic = make_critic(env);

%% Set Actor
actor = make_actor(env, pretrain_sample, mu, Sigma, actor_nits);

%% Set DDPG
agent = rlDDPGAgent(actor,critic);

agent.AgentOptions.NoiseOptions.StandardDeviation = zeros(bitshift(sample_dim, 1), 1);
agent.AgentOptions.ExperienceBufferLength=10^6;
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-6;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = min(size(Sigma, 1) / norm(Sigma, 'fro')^2, 1e-5);
agent.AgentOptions.ResetExperienceBufferBeforeTraining = true;

%% Training
trainOpts = rlTrainingOptions( ...
    "MaxEpisodes", 100, ...
    "StopTrainingCriteria", "EpisodeCount", ...
    "Plots", "none" ...
    );
trainingInfo = train(agent,env,trainOpts);

%% Save Store
save_store(env, 'train');
save("pretrain_sample.mat", "pretrain_sample", '-v7.3');
save("trainingInfo.mat", "trainingInfo", '-v7.3');

%% Simulation
initial_sample_sim = env.store_accepted_sample{end};
env_sim = RLMHEnv(log_target_pdf, initial_sample_sim, mu, Sigma);

simOptions = rlSimulationOptions(MaxSteps=5000);
experience = sim(env_sim, agent, simOptions);
save_store(env_sim, 'sim');
save("experience.mat", "experience", '-v7.3');
