parameters {
  real x;
}

model {
  target += log_mix(0.5,
                    normal_lpdf(x | -5, 1),
                    normal_lpdf(x | 5, 1));
}
