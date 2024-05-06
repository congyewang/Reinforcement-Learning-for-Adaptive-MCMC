% function [] = batch_load()
lib_super_path = "../../experiments/trails";

add_log_target_pdf_lib(lib_super_path);
add_lib_path(lib_super_path);

t = search_model_table(lib_super_path);

origin_pdb_condition = cellfun(@isempty, strfind(t.name, 'test_'));
origin_t = t(origin_pdb_condition, :);

sorted_origin_t = sortrows(origin_t, 'dim');

disp(sorted_origin_t);

% free_log_target_pdf_lib(lib_super_path);
% end
