import os
import json
import pandas as pd
import bridgestan as bs
from posteriordb import PosteriorDatabase

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from mcmctoolbox.functoolbox import flat
from rlmcmc_policy import RLMHTD3Policy, RLMHDDPG


# Load DataBase Locally
pdb_path = os.path.join(os.getcwd(), "../Python/posteriordb/posterior_database")
my_pdb = PosteriorDatabase(pdb_path)

# Load Dataset
posterior = my_pdb.posterior("test-AnnulusGaussianMixture-test-AnnulusGaussianMixture")
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
# grad_log_p = lambda x: model.log_density_gradient(x)[1]
# hess_log_p = lambda x: model.log_density_hessian(x)[2]


rl_mh_env = gym.make(
    "RLMHEnv-v0", log_target_pdf=log_p, sample_dim=2, total_timesteps=10_000
)


model = RLMHDDPG(RLMHTD3Policy, rl_mh_env, verbose=1, learning_starts=-1, device="cpu")

model.learn(total_timesteps=10_000, progress_bar=True)


state_list = np.array([i for i in rl_mh_env.store_state]).reshape(
    -1, rl_mh_env.sample_dim
)


# Trace Plot
plt.plot(
    state_list[:, 0], state_list[:, 1], "o-", linewidth=0.5, markersize=1.5, alpha=0.5
)
plt.title("Trace Plot")
plt.xlabel("Steps")
plt.ylabel("State")
plt.show()

# Acceptance Rate
print(np.sum(rl_mh_env.store_accetped_status) / len(rl_mh_env.store_accetped_status))

# Reward
plt.plot(np.array(rl_mh_env.store_reward))
