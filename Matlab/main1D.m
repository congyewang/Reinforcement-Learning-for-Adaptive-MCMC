clc;
clear;

% Create Env
env = MyEnv();

% Obtaining the information of observation and action from the environment
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Create an actor representation, which is a simple neural network from observation to action
statePath = [
    featureInputLayer(1,'Normalization','none','Name','State')
    fullyConnectedLayer(24,'Name','C')
    reluLayer('Name','relu1')
    fullyConnectedLayer(24,'Name','C2')
    % reluLayer('Name','relu2')
    eluLayer('Name','elu2')
    fullyConnectedLayer(1,'Name','Act') % !!!Make sure the name here matches the action name
    ];
actorNetwork = layerGraph(statePath);

% Defining Actor Options
actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);

% Defining Actor
actor = rlDeterministicActorRepresentation(actorNetwork, obsInfo, actInfo, 'Observation', {'State'}, 'Action', {'Act'}, actorOptions);

% Created by Critic to represent a simple fully connected neural network
statePath = [
    featureInputLayer(1,'Normalization','none','Name','State')
    fullyConnectedLayer(24,'Name','CState')
    reluLayer('Name','CReluState')];
actionPath = [
    featureInputLayer(1,'Normalization','none','Name','Action')
    fullyConnectedLayer(24,'Name','CAction')
    reluLayer('Name','CReluAction')];
commonPath = [
    additionLayer(2,'Name','add')
    % reluLayer('Name','CReluCommon')
    eluLayer('Name','CReluCommon')
    fullyConnectedLayer(1,'Name','QValue')];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);
criticNetwork = connectLayers(criticNetwork, 'CReluState', 'add/in1');
criticNetwork = connectLayers(criticNetwork, 'CReluAction', 'add/in2');

% Defining the Critic option
criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);

% Create a Critic representation
critic = rlQValueRepresentation(criticNetwork, obsInfo, actInfo, 'Observation',{'State'}, 'Action',{'Action'}, criticOpts);

% Defining options for Agent
agentOptions = rlDDPGAgentOptions('SampleTime', 1.0, 'DiscountFactor', 0.99, 'MiniBatchSize', 64);

% Create an Agent
agent = rlDDPGAgent(actor, critic, agentOptions);

% Set Training Options
trainOpts = rlTrainingOptions(...
    MaxEpisodes=10, ...
    MaxStepsPerEpisode=env.MaxSteps, ...
    ScoreAveragingWindowLength=5, ...
    Verbose=false, ...
    Plots='training-progress');

% Training
trainingStats = train(agent,env,trainOpts);

% Save
save('trained_agent.mat', 'agent');

% Load
% load('trained_agent.mat');
