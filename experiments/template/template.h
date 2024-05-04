#include "bridgestan.h"
#include <stdio.h>
#include <math.h>

int matlab_log_density_gradient(const char *data, const bool propto, const bool jacobian, const double *theta_unc, double *val, double *grad);

int matlab_param_num(const char *data, int *param_num);

int matlab_param_unc_num(const char *data, int *param_unc_num);

int matlab_param_unconstrain(const char *data, const double *theta, double *theta_unc);

int matlab_param_constrain(const char *data, const bool include_tp, const bool include_gq, const double *theta_unc, double *theta);
