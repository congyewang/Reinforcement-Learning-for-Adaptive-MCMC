import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import numpy as np

from scipy.stats import multivariate_normal
from scipy.special import expit
import logging

from mcmctoolbox.functoolbox import nearestPD

INF = 3.4028235e+38 # Corresponds to the value of FLT_MAX in C++


logging.basicConfig(level=logging.ERROR)


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
        self.action_space = spaces.Box(low=-INF, high=INF, shape=(1, 2 * sample_dim**2), dtype=np.float64)

        # Store
        self.store_state = []
        self.store_action = []
        self.store_log_accetance_rate = []
        self.store_accetped_status = []
        self.store_reward = []

        self._store_proposed_sample = []
        self._store_current_sample = []

        self._store_log_target_proposed = []
        self._store_log_target_current = []
        self._store_log_proposal_proposed = []
        self._store_log_proposal_current = []

        self._store_proposed_nearest_cov = []
        self._store_current_nearest_cov = []

    def log_proposal_pdf(self, x, mean, cov):
        """Multivariate normal distribution."""
        return multivariate_normal.logpdf(x, mean.flatten(), cov, allow_singular=False).squeeze()
        # return (-0.5 * len(mean) * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov)) - 0.5 * (x - mean) @ np.linalg.inv(cov) @ (x - mean).T).squeeze()

    def distance_function(self, current_sample, proposed_sample):
        return np.linalg.norm(current_sample - proposed_sample, 2)

    def reward_function(self, current_sample, proposed_sample, log_alpha):
        return (np.power(self.distance_function(current_sample, proposed_sample), 2) * np.exp(log_alpha)).flatten()[0]

    def step(self, action):
        # Extract current sample
        current_sample = self.state[:, 0:self.sample_dim]
        mcmc_noise = self.state[:, self.sample_dim:]

        # Extract Current Covariance, Proposed Sample, and Proposed Covariance
        current_cov_vec = action[:, 0:self.sample_dim**2]
        current_cov = current_cov_vec.reshape(self.sample_dim, self.sample_dim)

        proposed_cov_vec = action[:, self.sample_dim**2:]
        proposed_cov = proposed_cov_vec.reshape(self.sample_dim, self.sample_dim)

        # Avoid Singular Covariance
        nearest_cov_proposed = nearestPD(proposed_cov)
        nearest_cov_current = nearestPD(current_cov)

        L = np.linalg.cholesky(nearest_cov_current)

        # Generate Proposed Sample
        proposed_sample = current_sample + np.matmul(mcmc_noise, L)

        # Accept/Reject Process
        log_target_proposed = self.log_target_pdf(proposed_sample)
        log_target_current = self.log_target_pdf(current_sample)

        ## Avoid -np.inf
        if np.isneginf(log_target_proposed):
            log_target_proposed = -INF
        if np.isneginf(log_target_current):
            log_target_current = -INF

        try:
            log_proposal_proposed = self.log_proposal_pdf(proposed_sample, current_sample, nearest_cov_current)
        except Exception as e:
            print("nearest_cov_current:", nearest_cov_current)
            raise

        try:
            log_proposal_current = self.log_proposal_pdf(current_sample, proposed_sample, nearest_cov_proposed)
        except Exception as e:
            print("nearest_cov_proposed:", nearest_cov_proposed)
            raise

        log_alpha = min(0.0, log_target_proposed \
                - log_target_current \
                + log_proposal_current \
                - log_proposal_proposed)

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
        self.store_action.append(action)
        self.store_log_accetance_rate.append(log_alpha)

        self._store_proposed_sample.append(proposed_sample)
        self._store_current_sample.append(current_sample)

        self._store_log_target_proposed.append(log_target_proposed)
        self._store_log_target_current.append(log_target_current)
        self._store_log_proposal_proposed.append(log_proposal_proposed)
        self._store_log_proposal_current.append(log_proposal_current)

        self._store_proposed_nearest_cov.append(nearest_cov_proposed)
        self._store_current_nearest_cov.append(nearest_cov_current)

        # Calculate Reward
        reward = self.reward_function(current_sample, proposed_sample, log_alpha)
        self.store_reward.append(reward)

        # Check for Completion
        terminated = self._steps > self.total_timesteps
        truncated = terminated
        if terminated:
            pass

        # Update Iteration Time
        self.state = state
        self._steps += 1

        # Information
        info = {
            "state": state,
            "accepted_sample": accepted_sample,
            "current_sample": current_sample,
            "proposed_sample": proposed_sample,
            "accepted_status": accepted_status,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated
        }

        logging.info(f"Terminated: {terminated}")
        logging.info(f"Truncated: {truncated}")

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
        self.store_log_accetance_rate.append(0.0)

        # Information
        info = {
            "state": self.state,
            "accepted_status": True,
            "reward": 0.0
        }

        return self.state, info

    def render(self):
        pass

    def close(self):
        pass

class RLMHEnvA(RLMHEnv):
    def distance_function(self, current_sample, proposed_sample):
        return np.min(np.abs(current_sample - proposed_sample))

class RLMHEnvB(RLMHEnvA):
    def reward_function(self, current_sample, proposed_sample, log_alpha, weight=0.7, kappa=1.0):
        return (
                weight * expit(kappa * self.distance_function(current_sample, proposed_sample) * np.exp(log_alpha)) \
              + (1.0 - weight) * np.exp(self.log_target_pdf(proposed_sample))
            ).flatten()[0]
