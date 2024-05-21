parameters {
  real x;
}

model {
      if (x >= -6 && x <= -4) {
      target += log(0.25);
    } else if (x >= 4 && x <= 6) {
      target += log(0.25);
    } else {
      target += -1e+10;
    }
}
