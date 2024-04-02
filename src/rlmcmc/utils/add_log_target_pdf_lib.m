function add_log_target_pdf_lib(folder_path)

lib_path = search_lib(folder_path);
h_file_path = cell(size(lib_path));

for i = 1:length(lib_path)
    [path_str, lib_name, ~] = fileparts(lib_path{i});
    h_file_path{i} = fullfile(path_str, [lib_name '.h']);
end

for j = 1:length(lib_path)
    try
        [~, ~] = loadlibrary(lib_path{j}, h_file_path{j});
    catch ME
        disp('Load library error.');
        disp(ME.message);
        disp(lib_path{j});
    end

end

end
