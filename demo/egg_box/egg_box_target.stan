functions {
  real log_sum_exp_vector(vector x) {
    real max_val = max(x);
    return max_val + log(sum(exp(x - max_val)));
  }
}

parameters {
  vector[2] x;
}

model {
  vector[2] mu1;
  vector[2] mu2;
  vector[2] mu3;
  vector[2] mu4;
  matrix[2, 2] cov;
  vector[4] log_weights;

  mu1[1] = -5;
  mu1[2] = -5;
  mu2[1] = -5;
  mu2[2] =  5;
  mu3[1] =  5;
  mu3[2] = -5;
  mu4[1] =  5;
  mu4[2] =  5;

  cov = diag_matrix(rep_vector(1, 2));

  log_weights[1] = log(0.25) + multi_normal_lpdf(x | mu1, cov);
  log_weights[2] = log(0.25) + multi_normal_lpdf(x | mu2, cov);
  log_weights[3] = log(0.25) + multi_normal_lpdf(x | mu3, cov);
  log_weights[4] = log(0.25) + multi_normal_lpdf(x | mu4, cov);

  target += log_sum_exp_vector(log_weights);
}
