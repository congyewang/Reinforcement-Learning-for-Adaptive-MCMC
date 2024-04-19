function average_distance = expected_square_jump_distance(data)
    distances = vecnorm(diff(data, 1, 1), 2, 2);
    average_distance = mean(distances);
end