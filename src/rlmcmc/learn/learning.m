clear;
clc;
rng(0);

%% Set Env
env = RLMHEnvV3;

%% Set Critic
critic = make_critic(env);

%% Set Actor
actor = make_actor(env);

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
plot(cell2mat(env.store_reward));
title('Immediate Reward Plot');

figure;
plot(cumsum(cell2mat(env.store_reward)));
title('Cumulative Reward Plot');
