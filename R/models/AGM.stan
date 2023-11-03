data {
  int<lower=0> N;

  real<lower=0> w1;
  real<lower=0> w2;

  vector[N] mean;
  matrix[N, N] cov;

  real<lower=0> r0;
  real<lower=0> sigma;
}

parameters {
  vector[N] x;
}

model {
  target += log(
    w1 * exp(multi_normal_lpdf(x | mean, cov)) +
    w2 * exp(normal_lpdf(norm2(x) | r0, sigma) + log(if_else(norm2(x), 1, 0)) -  )
    );
}
