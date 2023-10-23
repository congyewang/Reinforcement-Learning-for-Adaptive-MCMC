functions {
  real log_p(matrix uniform_areas, vector x) {
    for (i in 1:8) {
      if (uniform_areas[i, 1] <= x[1] && x[1] <= uniform_areas[i, 2]
          && uniform_areas[i, 3] <= x[2] && x[2] <= uniform_areas[i, 4]) {
        return log(1.0 / 32);
      }
    }
    return negative_infinity();
  }
}

data {
  matrix[8, 4] uniform_areas;
}

parameters {
  vector[2] x;
}

model {
  target += log_p(uniform_areas, x);
}
