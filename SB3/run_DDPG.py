import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


from rl_mcmc_env.rl_mh import RLMHEnv
from rlmcmc_policy.policy import RLMHTD3Policy, RLMHTD3


rl_mh_env = gym.make('RLMHEnv-v0', log_target_pdf=lambda x: -x @ x.T / 2, sample_dim=2, total_timesteps=1000)

model = RLMHTD3(
    RLMHTD3Policy,
    rl_mh_env,
    verbose=1,
    learning_starts=-1,
    device='cpu')

model.learn(total_timesteps=1000, progress_bar=True)

# Diagnostic
state_list = np.array([i for i in rl_mh_env.store_state]).reshape(-1, rl_mh_env.sample_dim)

## Trace Plot
plt.plot(state_list[:, 0], state_list[:, 1], 'o-', linewidth=0.5, markersize=1.5, alpha=0.5)
plt.title("Trace Plot")
plt.xlabel("Steps")
plt.ylabel("State")
plt.show()

print("Acceptance Rate:", np.sum(rl_mh_env.store_accetped_status) / 1002)
