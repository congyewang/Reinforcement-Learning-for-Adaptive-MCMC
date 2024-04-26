#include "bridgestan.h"
#include <stdio.h>
#include <math.h>

int matlab_log_density_gradient(const char* data, const double *theta_unc, double *val, double *grad);

int matlab_param_num(const char* data, int *param_num);
