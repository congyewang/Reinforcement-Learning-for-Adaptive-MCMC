clearvars;
clc;
rng(1234);

%% Add Packages
% addpath(genpath('../../../src/rlmcmc/'));

%% Add Log Target PDF
share_name = "earnings_log10earn_height";
lib_super_path = "../../experiments/trails/earnings_log10earn_height";

add_log_target_pdf_lib(lib_super_path);
add_lib_path(lib_super_path);

%% Set Env
% Warm Start
warm_start_initial_sample = [0.0, 0.0, 0.0];
sample_dim = size(warm_start_initial_sample, 2);
[store, ~, Sigma, lambda] = am_mh(@log_target_pdf, 2000, warm_start_initial_sample);

env_initial_sample = store(end, :);
env_covariance = lambda(end) * Sigma(:, :, end);
env = RLMHEnv(@log_target_pdf, env_initial_sample, env_covariance);

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

%% Set Critic
% Define observation and action paths
obsPath = featureInputLayer(prod(obsInfo.Dimension), Name="obsInLyr");
actPath = featureInputLayer(prod(actInfo.Dimension), Name="actInLyr");
% Define common path: concatenate along first dimension
commonPath = [
    concatenationLayer(1,2,Name="concat")
    fullyConnectedLayer(int16(nearest_even(sample_dim)))
    reluLayer
    fullyConnectedLayer(1)
    ];
% Add paths to layerGraph network
criticNet = layerGraph(obsPath);
criticNet = addLayers(criticNet, actPath);
criticNet = addLayers(criticNet, commonPath);
% Connect paths
criticNet = connectLayers(criticNet,"obsInLyr","concat/in1");
criticNet = connectLayers(criticNet,"actInLyr","concat/in2");
% Create the critic
critic = rlQValueFunction(criticNet,obsInfo,actInfo,...
    ObservationInputNames="obsInLyr", ...
    ActionInputNames="actInLyr");

%% Set Actor
% Create a network to be used as underlying actor approximator
actorNet = [
    featureInputLayer(prod(obsInfo.Dimension))
    TwinNetworkLayer( ...
    'Name', 'twin_network_layer', ...
    'input_nodes', bitshift(prod(obsInfo.Dimension), -1), ...
    'hidden_nodes', int16(nearest_even(sample_dim) + 2), ...
    'output_nodes', bitshift(prod(actInfo.Dimension), -1), ...
    'mag', 1e-9);
    ];

% Convert to dlnetwork object
actorNet = dlnetwork(actorNet);

% Pre-Train
pre_train_targets_proposed = zeros(size(store));

for i = 1:size(store, 1)
    pre_train_targets_proposed(i, :) = mvlaprnd(sample_dim, store(i, :)', env_covariance);
end

pre_train_targets =  dlarray([store, pre_train_targets_proposed]);
pre_train_features = pre_train_targets;

pre_train_options = trainingOptions("adam");

pre_train_options.MaxEpochs = 100;
pre_train_options.Plots = "training-progress";
pre_trained_actorNet = trainnet( ...
    pre_train_features, ...
    pre_train_targets, ...
    actorNet, "mean-squared-error", ...
    pre_train_options ...
    );

% Create the actor
actor = rlContinuousDeterministicActor(pre_trained_actorNet,obsInfo,actInfo);
% actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);

%% Set DDPG
agent = rlDDPGAgent(actor,critic);
agent.AgentOptions.ExperienceBufferLength = 1e6;
agent.AgentOptions.NoiseOptions.StandardDeviation = zeros(actInfo.Dimension);

agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-2;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1e-2;

agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-6;
% agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1e-10;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1e-9;

%% Training
trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 200;
trainOpts.StopTrainingCriteria = "EpisodeCount";
trainOpts.Verbose = false;
% trainOpts.Plots = "none";
trainingInfo = train(agent,env,trainOpts);

%% Save Store
% save_store(env);
expected_square_jump_distance(cell2mat(env.store_accepted_sample)')

%% Log Target PDF
function [res] = log_target_pdf(x)

[res, ~] = wrapped_log_target_pdf(x', "earnings_log10earn_height");

end