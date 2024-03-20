function actor = make_actor(env)
%% Get Information
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

%% Set Critic
% Create a network to be used as underlying actor approximator
actorNet = [
    featureInputLayer(prod(obsInfo.Dimension))
    TwinNetworkLayer( ...
    'Name', 'twin_network_layer', ...
    'input_nodes', bitshift(prod(obsInfo.Dimension), -1), ...
    'hidden1_nodes', 256, ...
    'hidden2_nodes', 256, ...
    'output_nodes', bitshift(prod(actInfo.Dimension), -1));
    ];
% Convert to dlnetwork object
actorNet = dlnetwork(actorNet);
% Create the actor
actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);
end