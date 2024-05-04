#include "bridgestan.h"
#include <stdio.h>
#include <math.h>
#include <stdbool.h>


int matlab_log_density_gradient(const char *data, const bool propto, const bool jacobian, const double *theta_unc, double *val, double *grad) {
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

int matlab_param_unc_num(const char *data, int *param_unc_num) {
  char *err_model;
  int random_seed = 0;
  bs_model *model;

  model = bs_model_construct(data, random_seed, &err_model);
  if (!model) {
      if (err_model) {
          printf("Error: %s\n", err_model);
          bs_free_error_msg(err_model);
      }
      return -1;
  }

  *param_unc_num = bs_param_unc_num(model);

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

int matlab_param_constrain(const char *data, const bool include_tp, const bool include_gq, const double *theta_unc, double *theta) {
  char *err_model;
  int random_seed = 0;
  bs_model *model;
  bs_rng *rng;
  char *err_rng;
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

  rng = bs_rng_construct(random_seed, &err_rng);
  if (!rng) {
    if (err_rng) {
      printf("Error: %s\n", err_rng);
      bs_free_error_msg(err_rng);
    }
  return -1;
  }

  status = bs_param_constrain(model, include_tp, include_gq, theta_unc, theta, rng, &err_msg);
  if (err_msg) {
    printf("Error: %s\n", err_msg);
    bs_free_error_msg(err_msg);
    return -1;
  }

  bs_rng_destruct(rng);
  bs_model_destruct(model);

  return 0;
}
