function total_model_name = search_total_model_name(folder_path)

total_lib_path = search_lib(folder_path);

total_model_name = cell(size(total_lib_path));

for i = 1:length(total_lib_path)
    [model_path, ~, ~] = fileparts(total_lib_path{i});
    splitted_model_path = strsplit(model_path, '/');
    total_model_name{i} = splitted_model_path{end};
end

end
