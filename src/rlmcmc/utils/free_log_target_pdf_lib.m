function free_log_target_pdf_lib(folder_path)

lib_path = search_lib(folder_path);

for i = 1:length(lib_path)
    [~, lib_name, ~] = fileparts(lib_path{i});
    if libisloaded(lib_name)
        unloadlibrary(lib_name);
    else
        fprintf("%s library not loaded.\n", lib_name);
    end

end

end
