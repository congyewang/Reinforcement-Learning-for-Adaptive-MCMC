function [theta_unc] = wrapped_param_unconstrain(theta, model_name)

model_name_replace = strrep(model_name, '-', '_');
data_path = which(strcat(model_name_replace, ".json"));
data_path_char = char(data_path);

theta_unc_in = NaN(size(theta));
[~,~,~,theta_unc]=calllib(strcat("lib", model_name_replace),"matlab_param_unconstrain", data_path_char ,theta, theta_unc_in);

end
