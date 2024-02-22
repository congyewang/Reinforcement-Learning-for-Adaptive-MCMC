clear;
clc;
rng(0);

%% Set Env
env = Gauss1DV8;
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Define observation and action paths
obsPath = featureInputLayer(prod(obsInfo.Dimension),Name="obsInLyr");
actPath = featureInputLayer(prod(actInfo.Dimension),Name="actInLyr");

%% Set Critic 
% Define common path: concatenate along first dimension
commonPath = [
    concatenationLayer(1,2,Name="concat")
    fullyConnectedLayer(32)
    reluLayer
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

critic = rlQValueFunction(criticNet,obsInfo,actInfo,...
    ObservationInputNames="obsInLyr", ...
    ActionInputNames="actInLyr");

%% Set Actor
% Create a network to be used as underlying actor approximator
actorNet = [
    featureInputLayer(prod(obsInfo.Dimension))
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(prod(actInfo.Dimension))
    ];

% Convert to dlnetwork object
actorNet = dlnetwork(actorNet);

% Display the number of weights
summary(actorNet)

actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);
%% Set DDPG

agent = rlDDPGAgent(actor,critic);

agent.AgentOptions.SampleTime=env.Ts;
agent.AgentOptions.TargetSmoothFactor=1e-3;
agent.AgentOptions.ExperienceBufferLength=1e6;
agent.AgentOptions.DiscountFactor=0.99;
agent.AgentOptions.MiniBatchSize=32;

agent.AgentOptions.CriticOptimizerOptions.LearnRate=1e-5;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold=1;

agent.AgentOptions.ActorOptimizerOptions.LearnRate=1e-5;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold=1;

%% Training Session
trainOpts = rlTrainingOptions;

trainOpts.MaxEpisodes = 1e05;
trainOpts.MaxStepsPerEpisode = 1;
trainOpts.StopTrainingCriteria = "EpisodeCount";
trainOpts.StopTrainingValue = 1e05;
trainOpts.ScoreAveragingWindowLength = 5;

trainOpts.Verbose = true;
trainOpts.Plots = "training-progress";

trainingInfo = train(agent,env,trainOpts);

%% Plot Learning Trace
trace_plot(env);

%% Plot Policy
generatePolicyFunction(agent);
policy_plot();

%% Plot Reward
figure;
plot(cell2mat(env.StoreReward));
title('Immediate Reward Plot');

figure;
plot(cumsum(cell2mat(env.StoreReward)));
title('Cumulative Reward Plot');
