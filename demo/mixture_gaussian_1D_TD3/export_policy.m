clearvars;
clc;
rng(0);

%% Add Packages
addpath(genpath('../../src/rlmcmc/'));

%% Export
load_agent1 = load(['savedAgents/Agent',num2str(1,'%u'),'.mat']);
generatePolicyFunction(load_agent1.saved_agent,"MATFileName",'load_agentData1.mat');
policy1 = coder.loadRLPolicy("load_agentData1.mat");
policy_save(policy1, 'policy1', -10, 10);

load_agent20 = load(['savedAgents/Agent',num2str(20,'%u'),'.mat']);
generatePolicyFunction(load_agent20.saved_agent,"MATFileName",'load_agentData20.mat');
policy20 = coder.loadRLPolicy("load_agentData20.mat");
policy_save(policy20, 'policy20', -10, 10);

load_agent140 = load(['savedAgents/Agent',num2str(140,'%u'),'.mat']);
generatePolicyFunction(load_agent140.saved_agent,"MATFileName",'load_agentData140.mat');
policy140 = coder.loadRLPolicy("load_agentData140.mat");
policy_save(policy140, 'policy140', -10, 10);
