data {
  real<lower=0> sigma;
  real<lower=0> r;
}

transformed data {
  int K = 4;
  matrix[4, 2] modes;
  matrix[2, 2] cov;

  // Calculate modes
  real d = r * sigma;
  modes[1, 1] = d;
  modes[1, 2] = d;
  modes[2, 1] = -d;
  modes[2, 2] = d;
  modes[3, 1] = -d;
  modes[3, 2] = -d;
  modes[4, 1] = d;
  modes[4, 2] = -d;

  // Set covariances
  cov = diag_matrix(rep_vector(sigma, 2));
}

parameters {
  vector[2] x;
}

model {
  real pdf = 0;
  for (k in 1:K) {
    pdf += exp(multi_normal_lpdf(x | modes[k], cov));
  }
  target += log(pdf);
}
