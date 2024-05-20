parameters {
  real x;
}

model {
      if (x >= -5 && x <= -4) {
      target += log(0.5);
    } else if (x >= 4 && x <= 5) {
      target += log(0.5);
    } else {
      target += -1e+10;
    }
}
