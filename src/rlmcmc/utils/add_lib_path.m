function add_lib_path(total_lib_folder_path)

lib_path = search_lib(total_lib_folder_path);

for i = 1:length(lib_path)
    [lib_folder_path, ~, ~] = fileparts(lib_path{i});
    addpath(lib_folder_path);
end

end
