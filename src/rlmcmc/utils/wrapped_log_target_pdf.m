function [val,grad] = wrapped_log_target_pdf(theta, model_name)

model_name_replace = strrep(model_name, '-', '_');
data_path = which(strcat(model_name_replace, ".json"));
data_path_char = char(data_path);

grad_in = NaN(size(theta));
[~,~,~,val,grad]=calllib(strcat("lib", model_name_replace),"matlab_log_density_gradient", data_path_char ,theta, NaN, grad_in);

end