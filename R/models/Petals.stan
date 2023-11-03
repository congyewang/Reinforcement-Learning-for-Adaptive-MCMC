parameters {
  vector[2] x;
}

model {
  real r = sqrt(x[1]^2 + x[2]^2);
  real theta = atan2(x[2], x[1]);

  target += -0.5 * r^2 + log(2 + sin(5 * theta));
}
