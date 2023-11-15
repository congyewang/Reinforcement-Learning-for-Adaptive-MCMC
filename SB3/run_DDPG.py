import numpy as np
from torch import nn
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

from rlmcmc.env import RLMHEnv
from rlmcmc.policy import RLMHTD3Policy
from rlmcmc.agent import RLMHDDPG
from rlmcmc.toolbox.noise import RLMHNormalActionNoise

import json
import pandas as pd
import bridgestan as bs
from posteriordb import PosteriorDatabase

from mcmctoolbox.functoolbox import flat


# Load DataBase Locally
pdb_path = "/home/congye/Code/PythonProjects/LearningAdaptiveMCMC/Python/posteriordb/posterior_database"
my_pdb = PosteriorDatabase(pdb_path)

# Load Dataset
posterior = my_pdb.posterior("test-SimpleEggBox-test-SimpleEggBox")
stan = posterior.model.stan_code_file_path()
data = json.dumps(posterior.data.values())
model = bs.StanModel.from_stan_file(stan, data)

# Gold Standard
gs = posterior.reference_draws()
df = pd.DataFrame(gs)
gs_chains = np.zeros(
    (
        sum(flat(posterior.information["dimensions"].values())),
        posterior.reference_draws_info()["diagnostics"]["ndraws"],
    )
)
for i in range(len(df.keys())):
    s = []
    for j in range(len(df[df.keys()[i]])):
        s += df[df.keys()[i]][j]
    gs_chains[i, :] = s
linv = np.linalg.inv(np.cov(gs_chains))

# Extract log-P-pdf and its gradient
log_p = model.log_density


total_timesteps = 5_000
rlmh_env = gym.make(
    "RLMHEnv-v0", log_target_pdf=log_p, sample_dim=2, total_timesteps=total_timesteps
)


policy_kwargs = dict(
    net_arch=dict(pi=[64, 64, 64], qf=[64, 64, 64]),
    activation_fn=nn.Softplus,
    n_critics=1,
)


# The noise objects for DDPG
n_actions = rlmh_env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
)


model = RLMHDDPG(
    RLMHTD3Policy,
    rlmh_env,
    seed=1234,
    policy_kwargs=policy_kwargs,
    learning_starts=-1,
    batch_size=64,
    train_freq=(2, "step"),
    action_noise=action_noise,
    device="cpu",
    verbose=2,
)

model.learn(total_timesteps=total_timesteps, progress_bar=True)


state_list = np.array([i for i in rlmh_env.store_state]).reshape(
    -1, get_flattened_obs_dim(rlmh_env.observation_space)
)


plt.plot(state_list[:, 0], state_list[:, 1], "o-", alpha=0.1)
plt.show()
