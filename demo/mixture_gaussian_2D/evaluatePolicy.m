function action1 = evaluatePolicy(observation1)
%#codegen

% Reinforcement Learning Toolbox
% Generated on: 19-May-2024 23:38:54

persistent policy;
if isempty(policy)
	policy = coder.loadRLPolicy("load_agentData1401.mat");
end
% evaluate the policy
action1 = getAction(policy,observation1);