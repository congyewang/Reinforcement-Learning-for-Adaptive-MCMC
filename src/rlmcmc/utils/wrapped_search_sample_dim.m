function sample_dim = wrapped_search_sample_dim(model_name)

model_name_replace = strrep(model_name, '-', '_');
data_path = which(strcat(model_name_replace, ".json"));
data_path_char = char(data_path);

[~,~,sample_dim]=calllib(strcat("lib", model_name_replace),"matlab_param_num", data_path_char, 0);

end
