clearvars;
clc;
rng(1234);

%% Add Packages
% addpath("../actor")
% addpath("../agent")
% addpath("../env")
% addpath("../target")
% addpath("../utils")

%% Set Env
sample_dim = 3;
covariance = 2.38 * diag([0.03850445835832558, 8.556912195063501e-06, 6.330573874886414e-05]);
env = RLMHEnvV9(@log_target_pdf, sample_dim, covariance);

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

%% Set Critic
% Define observation and action paths
obsPath = featureInputLayer(prod(obsInfo.Dimension), Name="obsInLyr");
actPath = featureInputLayer(prod(actInfo.Dimension), Name="actInLyr");
% Define common path: concatenate along first dimension
commonPath = [
    concatenationLayer(1,2,Name="concat")
    fullyConnectedLayer(16)
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
        'hidden_nodes', 16, ...
        'output_nodes', bitshift(prod(actInfo.Dimension), -1));
    ];
% actorNet = [
%     featureInputLayer(prod(obsInfo.Dimension))
%     TwinNetworkLayerConstVar( ...
%         'Name', 'twin_network_layer', ...
%         'input_nodes', bitshift(prod(obsInfo.Dimension), -1), ...
%         'hidden_nodes', 16, ...
%         'output_nodes', bitshift(prod(actInfo.Dimension), -1));
%     ];

% Convert to dlnetwork object
actorNet = dlnetwork(actorNet);
% Display the number of weights
summary(actorNet)
% Create the actor
actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);

%% Set DDPG
agent = rlDDPGAgent(actor,critic);
agent.AgentOptions.ExperienceBufferLength=1e6;
agent.AgentOptions.NoiseOptions.StandardDeviation = zeros(actInfo.Dimension);
% agent.AgentOptions.CriticOptimizerOptions.L2RegularizationFactor=1e-2;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold=1;
agent.AgentOptions.ActorOptimizerOptions.L2RegularizationFactor=1e-2;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold=1e-8;

%% Training
trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 100;
%trainOpts.MaxStepsPerEpisode = 200;
trainOpts.StopTrainingCriteria = "EpisodeCount";
%trainOpts.Verbose = true;
%trainOpts.Plots = "training-progress";
trainOpts.SaveAgentCriteria = "EpisodeCount";
trainOpts.SaveAgentValue = 1;
trainOpts.SaveAgentDirectory = "savedAgents";
trainingInfo = train(agent,env,trainOpts);

%% Plot Learning Trace
trace_plot(env);

%% Plot Policy
% figure()
% plot_max = 3;
% for i = 1:plot_max
%     subplot(1,plot_max,i)
%     n = trainOpts.MaxEpisodes - i + 1;
%     load_agent = load(['savedAgents/Agent',num2str(n,'%u'),'.mat']);
%     generatePolicyFunction(load_agent.saved_agent,"MATFileName",'load_agentData.mat');
%     policy = coder.loadRLPolicy("load_agentData.mat");
%     delete("load_agentData.mat")
%     policy_plot_2D(policy);
% end

%% Plot Reward
reward_plot(env);
