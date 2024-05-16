functions {
  real log_sum_exp_vector(vector x) {
    real max_val = max(x);
    return max_val + log(sum(exp(x - max_val)));
  }
}

parameters {
  real x;
}

model {
  vector[5] log_weights;
  log_weights[1] = log(0.1) + normal_lpdf(x | -10, 1);
  log_weights[2] = log(0.2) + normal_lpdf(x | -5, 1);
  log_weights[3] = log(0.3) + normal_lpdf(x | -2, 1);
  log_weights[4] = log(0.2) + normal_lpdf(x | 2, 1);
  log_weights[5] = log(0.2) + normal_lpdf(x | 6, 1);

  target += log_sum_exp_vector(log_weights);
}
