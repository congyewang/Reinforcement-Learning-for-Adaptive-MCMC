parameters {
  real x;
}

model {
  target += skew_normal_lpdf(x | 0, 1, 2);
}
