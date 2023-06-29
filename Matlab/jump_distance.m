function distances = jump_distance(store)
distances = zeros(1, length(store) - 1);
for i = 2:length(store)
    distances(i) = norm(store(i-1,:) - store(i,:), 2)^2;
end
end
