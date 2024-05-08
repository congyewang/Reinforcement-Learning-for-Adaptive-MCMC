function save_store(env, mark)

if nargin < 2
    mark = 'train';
end

% Get a list of all properties
prop_list = properties(env);

% Loop through each property
for i = 1:length(prop_list)
    % Get the name of the current property
    prop_name = prop_list{i};

    % Check if the property name starts with 'store_'
    if startsWith(prop_name, 'store_')
        % Get the value of a property
        prop_value = env.(prop_name);
        data = cell2mat(prop_value)';

        % Build file name (property name.mat)
        filename = sprintf("%s_%s.mat", mark, prop_name);

        % Save to .mat file
        save(filename, 'data', '-v7.3');
    end
end

end
