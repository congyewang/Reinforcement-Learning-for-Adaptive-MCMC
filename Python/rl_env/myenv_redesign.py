import gymnasium as gym
from gymnasium import spaces

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

INF = 3.4028235e+38 # Corresponds to the value of FLT_MAX in C++
SEED = 1234
generator = np.random.Generator(np.random.PCG64(SEED))


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
        self.observation_space = spaces.Box(low=-INF, high=INF, shape=(2,))

        # Action specification
        self.action_space = spaces.Box(low=0, high=INF, shape=(2,))

        # Store
        self.store_state = []
        self.store_log_accetance_rate = []
        self.store_accetped_status = []
        self.store_action = []
        self.store_action_pair = []
        self.store_reward = []

    def step(self, action):
        theta_curr = self.state[0].reshape(self.dim,)
        theta_prop = theta_curr + action[0] * self.state[1].reshape(self.dim,)

        sigma_curr = action[0].reshape(self.dim,)
        sigma_prop = action[1].reshape(self.dim,)

        # Metropolis-Hastings
        log_p_prop = self.log_p(theta_prop)
        log_p_curr = self.log_p(theta_curr)
        log_q_prop = norm.logpdf(theta_prop, loc=theta_curr, scale=sigma_curr)
        log_q_curr = norm.logpdf(theta_curr, loc=theta_prop, scale=sigma_prop)

        log_alpha = min(
            log_p_prop \
            - log_p_curr \
            + log_q_curr \
            - log_q_prop,
            np.array([0.0])
            )

        self.store_log_accetance_rate.append(log_alpha)

        # Generate Epislon_{t+1}
        eps_next = generator.normal(loc=0, scale=1, size=(1,))

        self.store_action_pair.append([sigma_curr, sigma_prop])

        ## Accept or Reject
        if np.log(generator.uniform()) < log_alpha:
            accepted_status = True
            self.store_accetped_status.append(accepted_status)

            self.store_action.append(sigma_prop)

            theta_next = theta_prop
        else:
            accepted_status = False
            self.store_accetped_status.append(accepted_status)

            self.store_action.append(sigma_curr)

            theta_next = theta_curr

        # Calculate Rao Blackwell Reward
        norm_2 = np.linalg.norm(theta_curr.reshape(self.dim, -1) - theta_prop.reshape(self.dim, -1), 2)
        reward = np.power(norm_2, 2) * np.exp(log_alpha)
        self.store_reward.append(reward)

        # Update State and Iteration Time
        state_next = np.array([theta_next, eps_next]).flatten()
        self.store_state.append(state_next)
        self.state = state_next
        self.ts += 1

        # Check for Completion
        terminated = self.ts >= self.max_steps
        truncated = terminated
        if terminated:
            # self.reset()
            pass

        # Information
        info = {
            "state_next": state_next,
            "state_curr": theta_curr,
            "accepted_status": accepted_status,
            "reward": reward,
            "state_prop": theta_prop
        }

        return state_next, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        self.ts = 0
        self.state = np.array([
                10.0 * np.ones(self.dim),
                generator.normal(loc=0, scale=1, size=(1,))
            ]).flatten() # initialize s_{t}
        self.store_state.append(self.state)

        self.store_accetped_status.append(True)
        self.store_action.append(np.eye(self.dim))

        self.store_reward.append(np.array([0.0]))
        self.store_log_accetance_rate.append(np.array([0.0]))

        # Information
        info = {
            "state": self.state,
            "accepted_status": True,
            "reward": np.array([0.0])
        }

        return self.state, info

    def render(self, mode="human"):
        plt.plot(self.store_state)
        plt.show()
