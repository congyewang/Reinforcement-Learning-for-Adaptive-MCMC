#include "bridgestan.h"
#include <stdio.h>
#include <math.h>


int matlab_log_density_gradient(const double *theta_unc, double *val, double *grad) {
  char *data = "{{ data_path }}";
  char *err_model;
  int random_seed = 1234;
  bs_model *model;
  bool propto = true;
  bool jacobian = true;
  char *err_msg = NULL;
  int status;

  model = bs_model_construct(data, random_seed, &err_model);
  status = bs_log_density_gradient(model, propto, jacobian, theta_unc, val, grad, &err_msg);

  if (err_model) {
    printf("Error Model: %s\n", err_model);
    bs_free_error_msg(err_model);
  }

  if (err_msg) {
    printf("Error MSG: %s\n", err_msg);
    bs_free_error_msg(err_msg);
  }

  bs_model_destruct(model);

  return 0;
}
