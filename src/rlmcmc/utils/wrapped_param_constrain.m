function [theta] = wrapped_param_constrain(theta_unc, model_name)
include_tp = true;
include_gq = true;

model_name_replace = strrep(model_name, '-', '_');
data_path = which(strcat(model_name_replace, ".json"));
data_path_char = char(data_path);

theta_in = NaN(size(theta_unc));
[~,~,~,theta]=calllib(strcat("lib", model_name_replace),"matlab_param_constrain", data_path_char, include_tp, include_gq, theta_unc, theta_in);

end
