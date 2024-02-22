clear;
clc;
rng(0);

%% Set Env
env = Gauss1DV8;
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

%% Set Critic
criticBasisFcn = @(obs,act) [obs(1,1);
                             obs(2,1);
                             act(1,1);
                             act(2,1)];
% CriticW0 = rand(obsInfo.Dimension(1)+actInfo.Dimension(1),1);
CriticW0 = rand(4,1);
critic = rlQValueFunction({criticBasisFcn,CriticW0},obsInfo,actInfo);

%% Set Actor
actorBasisFcn = @(obs) [obs(1,1)
                        obs(2,1)];
% ActorW0 = rand(obsInfo.Dimension(1),actInfo.Dimension(1));
ActorW0 = rand(2,2);
actor = rlContinuousDeterministicActor({actorBasisFcn,ActorW0},obsInfo,actInfo);

%% Set DDPG
agent = rlDDPGAgent(actor,critic);

% agent.AgentOptions.SampleTime=env.Ts;
% agent.AgentOptions.TargetSmoothFactor=1e-3;
% agent.AgentOptions.ExperienceBufferLength=1e6;
% agent.AgentOptions.DiscountFactor=0.99;
% agent.AgentOptions.MiniBatchSize=32;
% 
% agent.AgentOptions.CriticOptimizerOptions.LearnRate=5e-3;
% agent.AgentOptions.CriticOptimizerOptions.GradientThreshold=1;
% 
% agent.AgentOptions.ActorOptimizerOptions.LearnRate=1e-4;
% agent.AgentOptions.ActorOptimizerOptions.GradientThreshold=1;

%% Training Session
trainOpts = rlTrainingOptions;

trainOpts.MaxEpisodes = 10000;
trainOpts.MaxStepsPerEpisode = 10;
trainOpts.StopTrainingCriteria = "EpisodeCount";
trainOpts.StopTrainingValue = 10000;
trainOpts.ScoreAveragingWindowLength = 5;

trainOpts.Verbose = true;
% trainOpts.Plots = "training-progress";

trainingInfo = train(agent,env,trainOpts);

%% Plot Learning Trace
trace_plot(env);

%% Plot Policy
generatePolicyFunction(agent);
policy_plot();

%% Plot Reward
plot(cell2mat(env.StoreReward));
