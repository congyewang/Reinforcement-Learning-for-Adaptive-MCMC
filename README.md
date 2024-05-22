# Reinforcement Learning for Adaptive MCMC
This code accompanies the paper "Reinforcement Learning for Adaptive MCMC". It was written in GCC v6.3.0, Matlab R2024a, and Python v3.10.13.

## Requirement
- Platform
  - Ubuntu 22.04.4 LTS x86_64
  - macOS 14.5 23F79 arm64
- Language
   - cmake v3.22.1
   - GCC v6.3.0
   - G++ v6.3.0
   - GNU Make v4.3
   - Matlab R2024a
   - Python v3.10.13
   - zsh v5.8.1
- Python Package
  - bridgestan v2.4.1
  - h5py v3.9.0
  - jinja2 v3.1.3
  - matplotlib v3.7.2
  - numpy v1.26.0
  - pandas v2.1.1
  - posteriordb v0.2.0
  - prettytable v3.10.0
  - pystan v3.9.1
  - pytorch v2.2.0
  - scipy v1.11.3
  - statsmodels v0.14.1
  - tqdm v4.65.0
  - toml v0.10.2
  - wget v3.2

Please note that this project has not been tested in Windows system, please consult the Stan, Bridgestan and Matlab documentation if needed.

## Demo
```{bash}
cd demo
cd <demo dir>
zsh run.sh
```

## Experiment
```{bash}
cd experiments
python main.py
zsh batch_run.sh "results" "learning.m"
zsh batch_baselines.sh
```
