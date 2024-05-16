parameters {
  real x;
}

model {
      if (x >= -4 && x <= -3) {
      target += log(0.25);
    } else if (x >= -1 && x <= 1) {
      target += log(0.5);
    } else if (x >= 3 && x <= 4) {
      target += log(0.25);
    } else {
      target += -1.7977e+308;
    }
}
