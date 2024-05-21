parameters {
  real x;
}

model {
  target += log_mix(0.6,
                    normal_lpdf(x | -5, 1),
                    normal_lpdf(x | 7, 2));
}
