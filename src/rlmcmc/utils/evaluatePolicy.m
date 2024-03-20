function action1 = evaluatePolicy(observation1)
%#codegen

% Reinforcement Learning Toolbox
% Generated on: 13-Mar-2024 20:22:36

persistent policy;
if isempty(policy)
	policy = coder.loadRLPolicy("load_agentData.mat");
end
% evaluate the policy
action1 = getAction(policy,observation1);