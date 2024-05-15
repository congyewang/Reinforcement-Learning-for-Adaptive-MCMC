clearvars;
clc;
rng({{ random_seed }});


%% Add Packages
addpath(genpath('../../../../src/rlmcmc/'));

%% Add Log Target PDF
lib_super_path = "../../../trails/{{ share_name }}";

add_log_target_pdf_lib(lib_super_path);
add_lib_path(lib_super_path);

%% add PosteriorDB
model_name = "{{ share_name }}";
sample_dim = wrapped_search_sample_dim(model_name);
log_target_pdf = @(x) wrapped_log_target_pdf(x',model_name);

%% Adaptive Metropolis
am_nits = 65000;
am_rate = {{ am_rate }};
stop_learning_iters = 60000;

[am_samples,~,~,~,am_rewards] = AdaptiveMetropolis(log_target_pdf, sample_dim, am_nits, am_rate, stop_learning_iters);

save("am_samples.mat", "am_samples", '-v7.3');
save("am_rewards.mat", "am_rewards", '-v7.3');
