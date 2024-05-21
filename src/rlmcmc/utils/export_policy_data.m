function export_policy_data(lb, ub)

if nargin < 1
    lb = -10;
end
if nargin < 2
    ub = 10;
end

for i=1:140

    load_agent = load(['savedAgents/Agent',num2str(i,'%u'),'.mat']);
    generatePolicyFunction(load_agent.saved_agent,"MATFileName",'load_agentData.mat');
    policy = coder.loadRLPolicy("load_agentData.mat");

    policy_save(policy, sprintf("policy%d", i), lb, ub);

    delete("load_agentData.mat");

end

end
