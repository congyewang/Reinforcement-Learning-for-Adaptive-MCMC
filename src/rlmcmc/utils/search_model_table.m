function [sucess_model_table] = search_model_table(folder_path)

total_model_name = search_total_model_name(folder_path);

sucess_model_name = {};
sucess_model_sample_dim = [];

for i = 1:length(total_model_name)
    try
        sucess_model_sample_dim(end+1) = wrapped_search_sample_dim(total_model_name{i});
        sucess_model_name{end+1} = total_model_name{i};
    catch ME
        disp('Load library error.');
        disp(ME.message);
        disp(total_model_name{i});
    end

end

name = sucess_model_name';
dim = sucess_model_sample_dim';

sucess_model_table = table(name, dim);

end
