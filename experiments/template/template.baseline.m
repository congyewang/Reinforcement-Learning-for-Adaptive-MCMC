clearvars;
clc;
rng(0);


%% Add Packages
addpath(genpath('../../../src/rlmcmc/'));

%% Add Log Target PDF
lib_super_path = "../../trails/{{ share_name }}";

add_log_target_pdf_lib(lib_super_path);
add_lib_path(lib_super_path);

%% add PosteriorDB
model_name = "{{ share_name }}";
sample_dim = wrapped_search_sample_dim(model_name);
log_target_pdf = @(x) wrapped_log_target_pdf(x',model_name);

%% Adaptive Metropolis
am_nits = {{ am_nits }};
am_rate = {{ am_rate }};

[am_samples,~,~,~] = AdaptiveMetropolis(log_target_pdf, sample_dim, am_nits, am_rate);

save("am_samples.mat", "am_samples");
