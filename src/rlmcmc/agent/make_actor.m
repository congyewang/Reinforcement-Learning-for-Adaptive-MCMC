function actor = make_actor( ...
    env, ...
    samples, ...
    actor_mean, ...
    actor_covariance, ...
    nits, ...
    neuron_num, ...
    max_epochs ...
    )

if nargin < 5
    nits = 100;
end
if nargin < 6
    neuron_num = 32;
end
if nargin < 7
    max_epochs = 2000;
end

if mod(neuron_num, 1) ~= 0
    error('The number of neurons must be an integer.');
end
if neuron_num <= 0
    error('The number of neurons must be greater than 0.');
end

%% Get Information
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

%% Set Actor
% Create a network to be used as underlying actor approximator
actorNet = [
    featureInputLayer(prod(obsInfo.Dimension))
    TwinNetworkLayer( ...
    'Name', 'twin_network_layer', ...
    'input_nodes', bitshift(prod(obsInfo.Dimension), -1), ...
    'hidden_nodes', neuron_num, ...
    'output_nodes', bitshift(prod(actInfo.Dimension), -1) ...
    );
    ];

% Convert to dlnetwork object
actorNet = dlnetwork(actorNet);

% Pretrain the actor
S_pretrain = [samples, samples];
phi_pretrain = [samples, samples];

% Divide targets into train and validate using random indices
[train_idx,val_idx,~] = dividerand(size(S_pretrain, 1),0.7,0.3,0.0);

S_pretrain_train = S_pretrain(train_idx', :);
phi_pretrain_train = phi_pretrain(train_idx', :);
S_pretrain_val = S_pretrain(val_idx', :);
phi_pretrain_val = phi_pretrain(val_idx', :);

pretrain_options = trainingOptions( ...
    "adam", ...
    "MaxEpochs", int16(max_epochs), ...
    'OutputFcn',@(info)stopIfValidationLossBelowOne(info), ...
    "OutputNetwork", "best-validation", ...
    "ValidationData", {S_pretrain_val, phi_pretrain_val} ...
    );

actorNet = trainnet(S_pretrain_train, ...
    phi_pretrain_train, ...
    actorNet, ...
    "mse", ...
    pretrain_options);

% Create the actor
actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);

end

function stop = stopIfValidationLossBelowOne(info)
    stop = false;
    if info.State == "iteration"
        if ~isempty(info.ValidationLoss) && info.ValidationLoss < 1
            disp('Validation loss is below 1, stopping training.')
            stop = true;
        end
    end
end
