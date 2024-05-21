function show_policy()

for i=1:140

load_agent = load(['savedAgents/Agent',num2str(i,'%u'),'.mat']);
generatePolicyFunction(load_agent.saved_agent,"MATFileName",'load_agentData.mat');
policy = coder.loadRLPolicy("load_agentData.mat");
policy_plot(policy, ["Step ", num2str(i), " Policy"], -10, 10);

delete("load_agentData.mat");
pause();

end

delete("load_agentData.mat");

end
