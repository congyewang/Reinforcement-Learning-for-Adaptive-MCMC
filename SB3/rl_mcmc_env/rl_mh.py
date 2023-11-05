import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import numpy as np

import logging

INF = 3.4028235e+38 # Corresponds to the value of FLT_MAX in C++


logging.basicConfig(level=logging.DEBUG)


class RLMHEnv(gym.Env):
    def __init__(self, log_target_pdf, sample_dim=2, total_timesteps=10_000):
        super().__init__()

        # Target Distribution
        self.log_target_pdf = log_target_pdf

        # Parameter
        self.sample_dim = sample_dim
        self.total_timesteps = total_timesteps
        self._steps = 0

        # Observation specification
        self.observation_space = spaces.Box(low=-INF, high=INF, shape=(1, 2*sample_dim), dtype=np.float64)

        # Action specification
        self.action_space = spaces.Box(low=0.0, high=INF, shape=(1, sample_dim+1), dtype=np.float64)

        # Store
        self.store_state = []
        self.store_log_accetance_rate = []
        self.store_accetped_status = []
        self.store_reward = []

    def step(self, action):
        logging.debug(f"No. {self._steps} Iteration")
        logging.debug(f"Current Action: {action}")

        # Extract current sample
        current_sample = self.state[:, 0:self.sample_dim]
        # Extract proposed sample and proposed density ratio
        proposed_sample, log_proposed_density_ratio = action[:, 0:self.sample_dim], action[:, self.sample_dim]

        # Accept/Reject Process
        log_alpha = self.log_target_pdf(proposed_sample) - self.log_target_pdf(current_sample) + log_proposed_density_ratio

        if np.log(self.np_random.uniform()) < log_alpha:
            accepted_status = True
            accepted_sample = proposed_sample
        else:
            accepted_status = False
            accepted_sample = current_sample

        # Update State
        mcmc_noise = self.np_random.normal(size=(1, self.sample_dim))
        state = np.hstack((accepted_sample, mcmc_noise))

        # Store
        self.store_state.append(state)
        self.store_accetped_status.append(accepted_status)
        # self.store_action.append(action)
        self.store_log_accetance_rate.append(log_alpha)

        # Calculate Reward
        reward = (np.power(np.linalg.norm(current_sample - proposed_sample, 2), 2) * np.exp(log_alpha)).flatten()[0]
        self.store_reward.append(reward)

        # Update Iteration Time
        self.state = state
        self._steps += 1

        # Check for Completion
        terminated = self._steps >= self.total_timesteps
        truncated = terminated
        if terminated:
            # self.reset()
            pass

        # Information
        info = {
            "state": state,
            "accepted_sample": accepted_sample,
            "current_sample": current_sample,
            "proposed_sample": proposed_sample,
            "accepted_status": accepted_status,
            "reward": reward
        }

        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Set Random Seed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # Initial Steps
        self._steps = 0

        # Initialize the state with [[0., 0., ..., 0.], [N(0, 1), N(0, 1), ..., N(0, 1)]
        self.state = np.hstack(
            (
                np.zeros((1, self.sample_dim)),
                self.np_random.normal(size=(1, self.sample_dim))
            )
        )

        # Store
        self.store_state.append(self.state)
        self.store_accetped_status.append(True)
        self.store_reward.append(0.0)
        self.store_log_accetance_rate.append(np.array([0.0]))

        # Information
        info = {
            "state": self.state,
            "accepted_status": True,
            "reward": 0.0
        }

        return self.state, info
