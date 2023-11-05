import numpy as np
import gymnasium as gym

from stable_baselines3 import DDPG

import rl_mcmc_env
from rlmcmc_policy.policy import RLMHTD3Policy


rl_mh_env = gym.make('RLMHEnv-v0', log_target_pdf=lambda x: -x @ x.T / 2, sample_dim=2)

model = DDPG(
    RLMHTD3Policy,
    rl_mh_env,
    verbose=1,
    learning_starts=1,
    device='cpu')

model.learn(total_timesteps=15)
