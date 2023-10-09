functions {
  real norm(vector x) {
    real result = 0;
    for (i in 1:size(x)) {
      result += x[i]^2;
    }
    return sqrt(result);
  }
}

data {
  int<lower=0> N;
  real<lower=0> r0;
  real<lower=0> sigma;
}

parameters {
  vector[N] x;
}

model {
  target += normal_lpdf(norm(x) | r0, sigma);
  target += normal_lpdf(x | 0, 1);
}
