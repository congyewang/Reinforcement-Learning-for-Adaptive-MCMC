clear all;
clc;
rng(0);

%% Set Env
env = Gauss1DV3;
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

%% Set Critic
% Define observation and action paths
obsPath = featureInputLayer(prod(obsInfo.Dimension), Name="obsInLyr");
actPath = featureInputLayer(prod(actInfo.Dimension), Name="actInLyr");
% Define common path: concatenate along first dimension
commonPath = [
    concatenationLayer(1,2,Name="concat")
    fullyConnectedLayer(8)
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
    TwinNetworkLayerV3( ...
        'Name', 'twin_network_layer', ...
        'input_nodes', bitshift(prod(obsInfo.Dimension), -1), ...
        'hidden1_nodes', 8, ...
        'hidden2_nodes', 8, ...
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

%% Training 
trainOpts = rlTrainingOptions;
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
