data {
  real<lower=0> mu;       // known mean
  real<lower=0> sigma2;   // known variance
}
parameters {
  vector[2] x;
}
model {
  target += -x[1]^2 / (2 * sigma2); // first term of the log probability
  target += -((x[2] - (x[1]^2 + mu))^2) / (2 * sigma2); // second term of the log probability
}
