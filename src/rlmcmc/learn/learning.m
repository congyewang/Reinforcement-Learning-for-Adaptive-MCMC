clearvars;
clc;
rng(0);

%% add PosteriorDB
model_name = 'earnings_log10earn_height';
sample_dim = wrapped_search_sample_dim(model_name);
log_target_pdf = @(x) wrapped_log_target_pdf(x',model_name);

%% get (approx) mean (mu) and covariance (Sigma) from adaptive mcmc
am_nits = 1000;
am_rate = 0.5;

[x_AMH,~,~,~] = AdaptiveMetropolis(log_target_pdf, sample_dim, am_nits, am_rate);

mu = mean(x_AMH(:,ceil(am_nits/2):end),2);
Sigma = cov(x_AMH(:,ceil(am_nits/2):end)');

n_pretrain = am_nits - ceil(am_nits/2) + 1;
X_pretrain = x_AMH(:,ceil(am_nits/2):end)'; % data for pre-training

%% Set environment - use (approx) mean (mu) and covariance (Sigma) to inform proposal
env = RLMHEnv(log_target_pdf,rand(1,3),mu,Sigma);
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
    TwinNetworkLayer( ...
        'Name', 'twin_network_layer', ...
        'input_nodes', prod(obsInfo.Dimension) / 2, ...
        'hidden_nodes', 32, ...
        'output_nodes', prod(actInfo.Dimension) / 2);
    ];

% Convert to dlnetwork object
actorNet = dlnetwork(actorNet);

% Pretrain the actor
S_pretrain = [X_pretrain, X_pretrain];
Sig_half = sqrtm(Sigma);
% inv_Sig_half = inv(Sig_half);
phi_pretrain = [(repmat(mu',n_pretrain,1) - X_pretrain) / Sig_half, ...
                (repmat(mu',n_pretrain,1) - X_pretrain) / Sig_half];
pretrain_options = trainingOptions("adam");
pretrain_options.MaxEpochs = 1000;
actorNet = trainnet(S_pretrain, ...
                    phi_pretrain, ...
                    actorNet, ...
                    "mse", ...
                    pretrain_options);
actorNet_predict = predict(actorNet,S_pretrain);
figure()
for i = 1:3
    subplot(1,3,i)
    plot(S_pretrain(:,i),actorNet_predict(:,i),'.')
    xlabel('x')
    ylabel(['phi_',num2str(i,'%u'),'(x)'])
    title('Pretrained Policy')
end

% Create the actor
actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);

%% Set DDPG
agent = rlDDPGAgent(actor,critic);

agent.AgentOptions.NoiseOptions.StandardDeviation = zeros(actInfo.Dimension);
agent.AgentOptions.ExperienceBufferLength=10^6;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold=1e-10;
agent.AgentOptions.ResetExperienceBufferBeforeTraining = true;

%% Training 
trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 100;
trainOpts.StopTrainingCriteria = "EpisodeCount";
trainOpts.Verbose = 1;
trainingInfo = train(agent,env,trainOpts);

%% Plot Learning Trace
trace_plot(env);

%% Plot Policy
actor = getActor(agent);
actorNet = getModel(actor);
actorNet_predict = predict(actorNet,S_pretrain);
figure()
for i = 1:3
    subplot(1,3,i)
    plot(S_pretrain(:,i),actorNet_predict(:,i),'.')
    xlabel('x')
    ylabel(['phi_',num2str(i,'%u'),'(x)'])
    title('Final Policy')
end
