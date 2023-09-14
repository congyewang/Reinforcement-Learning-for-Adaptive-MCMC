import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

from rlax import add_gaussian_noise

from base_rl_mcmc.toolbox import env_mh, INF, SEED


class MyEnv(gym.Env):
    def __init__(self, log_p, dim=1, max_steps=10_000):
        super(MyEnv, self).__init__()

        # Target Distribution
        self.log_p = log_p

        # Parameter
        self.dim = dim
        self.max_steps = max_steps
        self.ts = 0  # iteration time

        # Observation specification
        self.observation_space = spaces.Box(low=-INF, high=INF, shape=(dim,))

        # Action specification
        self.action_space = spaces.Box(low=-INF, high=INF, shape=(dim**2,))

        # Store
        self.store_state = [np.zeros(dim)]
        self.accetped_status = [True]
        self.store_action = [np.eye(dim)]
        self.store_reward = [0.0]

    def step(self, action, policy_func, noise_policy_func):
        sigma_curr = action

        # MCMC Environment
        state_curr, accepted_status, alpha, om = env_mh(theta_curr=self.state, sigma_curr=sigma_curr, policy_func=policy_func, noise_policy_func=noise_policy_func, log_p=self.log_p)

        # Store
        self.store_state.append(state_curr)
        self.accetped_status.append(accepted_status)
        self.store_action.append(sigma_curr)

        # Calculate Reward
        reward = np.power(np.linalg.norm(self.state - state_curr, 2), 2)
        self.store_reward.append(reward)

        # Update Iteration Time
        self.store = state_curr
        self.ts += 1

        # Check for Completion
        terminated = self.ts >= self.max_steps
        truncated = terminated
        if terminated:
            # self.reset()
            pass

        # Information
        info = {
            "state": state_curr,
            "accepted_status": accepted_status,
            "reward": reward,
            "alpha": alpha,
            "omega": om
        }

        return state_curr, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        self.ts = 0
        self.state = 10.0 * np.ones(self.dim)  # initialize s_{t}
        self.store_state.append(self.state)
        self.accetped_status.append(True)
        self.store_action.append(np.eye(self.dim))
        self.store_reward.append(0.0)

        # Information
        info = {
            "state": self.state,
            "accepted_status": True,
            "reward": 0.0,
            "alpha": 1.0,
            "omega": 1.0
        }
        return self.state, info

    def render(self, mode="human"):
        plt.plot(self.store_state)
        plt.show()
