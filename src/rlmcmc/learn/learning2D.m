clearvars;
clc;
rng(0);

%% Add Packages
addpath("../actor")
addpath("../agent")
addpath("../env")
addpath("../target")
addpath("../utils")

%% Set Env
env = RLMHEnvV11(@mixture_gaussian_target, 2);

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

%% Set Critic
% Define observation and action paths
obsPath = featureInputLayer(prod(obsInfo.Dimension), Name="obsInLyr");
actPath = featureInputLayer(prod(actInfo.Dimension), Name="actInLyr");
% Define common path: concatenate along first dimension
commonPath = [
    concatenationLayer(1,2,Name="concat")
    fullyConnectedLayer(32)
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
% actorNet = [
%     featureInputLayer(prod(obsInfo.Dimension))
%     TwinNetworkLayer( ...
%         'Name', 'twin_network_layer', ...
%         'input_nodes', bitshift(prod(obsInfo.Dimension), -1), ...
%         'hidden_nodes', 8, ...
%         'output_nodes', bitshift(prod(actInfo.Dimension), -1));
%     ];
actorNet = [
    featureInputLayer(prod(obsInfo.Dimension))
    TwinNetworkLayerConstVar( ...
        'Name', 'twin_network_layer', ...
        'input_nodes', bitshift(prod(obsInfo.Dimension), -1), ...
        'hidden_nodes', 8, ...
        'output_nodes', bitshift(prod(actInfo.Dimension), -1));
    ];

% Convert to dlnetwork object
actorNet = dlnetwork(actorNet);
% Display the number of weights
summary(actorNet)
% Create the actor
actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);

%% Set DDPG
agent = rlDDPGAgent(actor,critic);
%agent.AgentOptions.SampleTime=env.Ts;
%agent.AgentOptions.TargetSmoothFactor=1e-3;
%agent.AgentOptions.ExperienceBufferLength=1e6;
%agent.AgentOptions.DiscountFactor=0.99;
agent.AgentOptions.NoiseOptions.StandardDeviation = zeros(actInfo.Dimension);
%agent.AgentOptions.MiniBatchSize=64;
%agent.AgentOptions.CriticOptimizerOptions.LearnRate=1e-7;
% agent.AgentOptions.CriticOptimizerOptions.GradientThreshold=1;
%agent.AgentOptions.ActorOptimizerOptions.LearnRate=1e-7;
% agent.AgentOptions.ActorOptimizerOptions.GradientThreshold=0.00000001;

%% Training
trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 100;
%trainOpts.MaxStepsPerEpisode = 200;
trainOpts.StopTrainingCriteria = "EpisodeCount";
%trainOpts.StopTrainingValue = 1e05;
%trainOpts.ScoreAveragingWindowLength = 50;
%trainOpts.Verbose = true;
%trainOpts.Plots = "training-progress";
trainOpts.SaveAgentCriteria = "EpisodeCount";
trainOpts.SaveAgentValue = 1;
trainOpts.SaveAgentDirectory = "savedAgents";
%trainOpts.Plots = "none";
trainOpts.Verbose = 1;
trainingInfo = train(agent,env,trainOpts);

%% Plot Learning Trace
trace_plot(env);

%% Plot Policy
figure()
plot_max = 3;
for i = 1:plot_max
    subplot(1,plot_max,i)
    n = trainOpts.MaxEpisodes - i + 1;
    load_agent = load(['savedAgents/Agent',num2str(n,'%u'),'.mat']);
    generatePolicyFunction(load_agent.saved_agent,"MATFileName",'load_agentData.mat');
    policy = coder.loadRLPolicy("load_agentData.mat");
    delete("load_agentData.mat")
    policy_plot_2D(policy);
end

%% Plot Reward
reward_plot(env);
 