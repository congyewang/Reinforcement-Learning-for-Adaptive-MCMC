function action1 = evaluatePolicy(observation1)
%#codegen

% Reinforcement Learning Toolbox
% Generated on: 2024-02-22 12:25:23

persistent policy;
if isempty(policy)
	policy = coder.loadRLPolicy("agentData4.mat");
end
% evaluate the policy
action1 = getAction(policy,observation1);