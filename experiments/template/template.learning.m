clearvars;
clc;
rng(0);

%% Add Packages
addpath(genpath('../../../src/rlmcmc/'));

%% Add Log Target PDF
lib_super_path = "../../trails/{{ share_name }}";

add_log_target_pdf_lib(lib_super_path);
add_lib_path(lib_super_path);

%% add PosteriorDB
model_name = "{{ share_name }}";
sample_dim = wrapped_search_sample_dim(model_name);
log_target_pdf = @(x) wrapped_log_target_pdf(x',model_name);

%% get (approx) mean (mu) and covariance (Sigma) from adaptive mcmc
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
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold={{ gradient_clipping }};
agent.AgentOptions.ResetExperienceBufferBeforeTraining = true;

%% Training
trainOpts = rlTrainingOptions( ...
    "MaxEpisodes", 100, ...
    "StopTrainingCriteria", "EpisodeCount", ...
    "Plots", "none" ...
    );
trainingInfo = train(agent,env,trainOpts);

%% Save Store
save_store(env);
save("pretrain_sample.mat", "pretrain_sample");
save("actor.mat", "actor");
