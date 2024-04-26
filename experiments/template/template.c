#include "bridgestan.h"
#include <stdio.h>
#include <math.h>


int matlab_log_density_gradient(const char *data, const double *theta_unc, double *val, double *grad) {
  char *err_model;
  int random_seed = 0;
  bs_model *model;
  bool propto = true;
  bool jacobian = true;
  char *err_msg = NULL;
  int status;

  model = bs_model_construct(data, random_seed, &err_model);
  if (!model) {
      if (err_model) {
          printf("Error: %s\n", err_model);
          bs_free_error_msg(err_model);
      }
      return -1;
  }

  status = bs_log_density_gradient(model, propto, jacobian, theta_unc, val, grad, &err_msg);
  if (err_msg) {
      printf("Error: %s\n", err_msg);
      bs_free_error_msg(err_msg);
  }

  bs_model_destruct(model);

  return 0;
}

int matlab_param_num(const char *data, int *param_num) {
  char *err_model;
  int random_seed = 0;
  bs_model *model;

  bool include_tp = true;
  bool include_gq = true;

  model = bs_model_construct(data, random_seed, &err_model);
  if (!model) {
      if (err_model) {
          printf("Error: %s\n", err_model);
          bs_free_error_msg(err_model);
      }
      return -1;
  }

  *param_num = bs_param_num(model, include_tp, include_gq);

  bs_model_destruct(model);

  return 0;
}

int matlab_param_unconstrain(const char *data, const double *theta, double *theta_unc) {
  char *err_model;
  int random_seed = 0;
  bs_model *model;
  char *err_msg = NULL;
  int status;

  model = bs_model_construct(data, random_seed, &err_model);
  if (!model) {
      if (err_model) {
          printf("Error: %s\n", err_model);
          bs_free_error_msg(err_model);
      }
      return -1;
  }

  status = bs_param_unconstrain(model, theta, theta_unc, &err_msg);
  if (err_msg) {
      printf("Error: %s\n", err_msg);
      bs_free_error_msg(err_msg);
  }

  bs_model_destruct(model);

  return 0;
}
