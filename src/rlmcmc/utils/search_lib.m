function [lib_path] = search_lib(folder_path)

% Store Library Path
lib_path = {};

% Recursive Function for Traversing Directories and Subdirectories
    function search_folder(current_folder_path)
        items = dir(current_folder_path);

        for i = 1:length(items)
            if strcmp(items(i).name, '.') || strcmp(items(i).name, '..')
                continue;
            end

            current_item_path = fullfile(items(i).folder, items(i).name);

            if items(i).isdir
                search_folder(current_item_path);
            elseif strcmpi(items(i).name(end-2:end), '.so')
                [path_str, name, ~] = fileparts(current_item_path);
                current_item_path = fullfile(path_str, name);

                lib_path{end+1} = current_item_path;
            end
        end
    end

% Start Recursive Search
search_folder(folder_path);

end
