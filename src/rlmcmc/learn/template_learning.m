clearvars;
clc;
rng(0);

%% add PosteriorDB
model_name = "earnings_log10earn_height";
sample_dim = wrapped_search_sample_dim(model_name);
log_target_pdf = @(x) wrapped_log_target_pdf(x',model_name);

%% get (approx) mean (mu) and covariance (Sigma) from adaptive mcmc
am_nits = 1000;
am_rate = 0.5;

[x_AMH,~,~,~] = AdaptiveMetropolis(log_target_pdf, sample_dim, am_nits, am_rate);

mu = mean(x_AMH(:,ceil(am_nits/2):end),2);
Sigma = cov(x_AMH(:,ceil(am_nits/2):end)');

actor_nits = am_nits - ceil(am_nits/2) + 1;
pretrain_sample = x_AMH(:,ceil(am_nits/2):end)'; % data for pre-training

%% Set environment - use (approx) mean (mu) and covariance (Sigma) to inform proposal
env = RLMHEnv(log_target_pdf, rand(1, sample_dim), mu, Sigma);

%% Set Critic
critic = make_critic(env);

%% Set Actor
actor = make_actor(env, pretrain_sample, mu, Sigma, actor_nits);

%% Set DDPG
agent = rlDDPGAgent(actor,critic);

agent.AgentOptions.NoiseOptions.StandardDeviation = zeros(bitshift(sample_dim, 1), 1);
agent.AgentOptions.ExperienceBufferLength=10^6;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold=1e-10;
agent.AgentOptions.ResetExperienceBufferBeforeTraining = true;

%% Training
trainOpts = rlTrainingOptions( ...
    "MaxEpisodes", 100, ...
    "StopTrainingCriteria", "EpisodeCount", ...
    "Plots", "none" ...
    );
trainingInfo = train(agent,env,trainOpts);

%% Plot Learning Trace
trace_plot(env);

%% Save Store
save_store(env);
save("pretrain_sample.mat", "pretrain_sample");
save("actor.mat", "actor");
