function [val,grad] = wrapped_log_target_pdf(theta, data_path)

data_path_char = char(data_path);
[~, model_name, ~] = fileparts(data_path);

grad_in = NaN(size(theta));
[~,~,~,val,grad]=calllib(strcat("lib", model_name),"matlab_log_density_gradient", data_path_char ,theta, [NaN], grad_in);

end
