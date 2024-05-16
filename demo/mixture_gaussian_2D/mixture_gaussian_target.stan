parameters {
  vector[2] x;
}

model {
  vector[2] mu1;
  vector[2] mu2;
  matrix[2, 2] cov;
  vector[5] log_weights;

  mu1[1] = -3;
  mu1[2] = 0;
  mu2[1] = 3;
  mu2[2] = 0;
  cov = diag_matrix(rep_vector(1, 2));

  target += log_mix(0.5,
                    multi_normal_lpdf(x | mu1, cov),
                    multi_normal_lpdf(x | mu2, cov));
}
