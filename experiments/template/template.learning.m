clearvars;
clc;
rng(1234);

%% Add Packages
addpath(genpath('../../../src/rlmcmc/'));

%% Add Log Target PDF
lib_super_path = "../../trails/";

add_log_target_pdf_lib(lib_super_path);
add_lib_path(lib_super_path);

%% Set Env
initial_sample = {{ initial_sample }};
sample_dim = size(initial_sample, 2);
covariance = (2.38 / sqrt(sample_dim)) * {{ covariance_matrix }};
env = RLMHEnvV9(@log_target_pdf, initial_sample, covariance);

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

%% Set Critic
% Define observation and action paths
obsPath = featureInputLayer(prod(obsInfo.Dimension), Name="obsInLyr");
actPath = featureInputLayer(prod(actInfo.Dimension), Name="actInLyr");
% Define common path: concatenate along first dimension
commonPath = [
    concatenationLayer(1,2,Name="concat")
    fullyConnectedLayer({{ critic_hidden_units }})
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
    'hidden_nodes', {{ actor_hidden_units }}, ...
    'output_nodes', bitshift(prod(actInfo.Dimension), -1), ...
    'mag', 1e-9);
    ];

% Convert to dlnetwork object
actorNet = dlnetwork(actorNet);
% Create the actor
actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);

%% Set DDPG
agent = rlDDPGAgent(actor,critic);
agent.AgentOptions.ExperienceBufferLength=1e6;
agent.AgentOptions.NoiseOptions.StandardDeviation = zeros(actInfo.Dimension);

agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-2;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1e-2;

agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-6;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1e-10;

%% Training
trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 200;
trainOpts.StopTrainingCriteria = "EpisodeCount";
trainOpts.Verbose = false;
trainOpts.Plots = "none";
trainingInfo = train(agent,env,trainOpts);

%% Save Store
save_store(env);

%% Log Target PDF
function [res] = log_target_pdf(x)

[res, ~] = wrapped_log_target_pdf(x', '{{ share_name }}');

end