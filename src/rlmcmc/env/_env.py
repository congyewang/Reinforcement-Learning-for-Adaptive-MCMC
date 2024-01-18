import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal
from scipy.special import expit
from ..utils import Toolbox

from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Union, Dict, Any

INF = 3.4028235e38  # Corresponds to the value of FLT_MAX in C++


class RLMHEnvBase(gym.Env, ABC):
    def __init__(
        self,
        log_target_pdf: Callable[[NDArray[np.float32]], np.float32],
        sample_dim: int = 2,
        total_timesteps: int = 10_000,
    ) -> None:
        super().__init__()

        # Target Distribution
        self.log_target_pdf = log_target_pdf

        # Parameter
        self.sample_dim = sample_dim
        self.total_timesteps = total_timesteps
        self.steps: int = 0

        # Observation specification
        self.observation_space = spaces.Box(
            low=-INF, high=INF, shape=(2 * sample_dim,), dtype=np.float32
        )

        # Action specification
        self.action_space = spaces.Box(
            low=-INF, high=INF, shape=(2 * sample_dim**2,), dtype=np.float32
        )

        # Store
        self.store_observation: List[NDArray[np.float32]] = []
        self.store_action: List[NDArray[np.float32]] = []
        self.store_log_accetance_rate: List[np.float32] = []
        self.store_accetped_status: List[bool] = []
        self.store_reward: List[np.float32] = []

    def log_proposal_pdf(
        self,
        x: NDArray[np.float32],
        mean: NDArray[np.float32],
        cov: NDArray[np.float32],
    ) -> np.float32:
        return multivariate_normal.logpdf(x, mean.flatten(), cov)

    @abstractmethod
    def distance_function(
        self, current_sample: NDArray[np.float32], proposed_sample: NDArray[np.float32]
    ) -> np.float32:
        raise NotImplementedError("distance_function is not implemented.")

    @abstractmethod
    def reward_function(
        self,
        current_sample: NDArray[np.float32],
        proposed_sample: NDArray[np.float32],
        log_alpha: np.float32,
    ) -> np.float32:
        raise NotImplementedError("reward_function is not implemented.")

    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[
        NDArray[np.float32],
        np.float32,
        bool,
        bool,
        Dict[str, Union[NDArray[np.float32], bool, np.float32]],
    ]:
        # Check Action Shape
        assert (
            action.shape[0] == 2 * self.sample_dim**2 and action.shape[0] % 2 == 0
        ), f"Action shape is {action.shape}, but expected shape is {(2 * self.sample_dim**2, )}"

        # Extract Current Sample
        current_sample, mcmc_noise = np.split(
            self.observation, [self.sample_dim], axis=0
        )

        # Extract Flat Current Covariance and Flat Proposed Covariance
        current_vector, proposed_vector = np.split(action, [self.sample_dim**2], axis=0)

        # Reshape to Covariance Matrix
        current_covariance = current_vector.reshape(self.sample_dim, self.sample_dim)
        proposed_covariance = proposed_vector.reshape(self.sample_dim, self.sample_dim)

        # Avoid Singular Covariance
        nearest_proposed_covariance: NDArray[np.float32] = cov_nearest(proposed_covariance)
        nearest_current_covariance: NDArray[np.float32] = cov_nearest(current_covariance)

        # Generate Proposed Sample
        proposed_sample = current_sample + np.matmul(
            mcmc_noise, np.linalg.cholesky(nearest_current_covariance)
        )

        # Calculate Log Target Density
        log_target_proposed = self.log_target_pdf(proposed_sample)
        log_target_current = self.log_target_pdf(current_sample)

        ## Avoid -np.inf
        if np.isneginf(log_target_proposed):
            log_target_proposed = -INF
        if np.isneginf(log_target_current):
            log_target_current = -INF

        # Calculate Log Proposal Densitys
        log_proposal_proposed = self.log_proposal_pdf(
            proposed_sample, current_sample, nearest_current_covariance
        )
        log_proposal_current = self.log_proposal_pdf(
            current_sample, proposed_sample, nearest_proposed_covariance
        )

        # Calculate Log Acceptance Rate
        log_alpha = np.min(
            [
                0.0,
                log_target_proposed
                - log_target_current
                + log_proposal_current
                - log_proposal_proposed,
            ]
        )

        # Accept or Reject
        if np.log(self.np_random.uniform()) < log_alpha:
            accepted_status = True
            accepted_sample = proposed_sample
        else:
            accepted_status = False
            accepted_sample = current_sample

        # Update Observation
        next_mcmc_noise = self.np_random.normal(size=self.sample_dim)
        self.observation = np.hstack((accepted_sample, next_mcmc_noise))

        # Store
        self.store_observation.append(self.observation)
        self.store_accetped_status.append(accepted_status)
        self.store_action.append(action)
        self.store_log_accetance_rate.append(log_alpha)

        # Calculate Reward
        reward = self.reward_function(current_sample, proposed_sample, log_alpha)
        self.store_reward.append(reward)

        # Check for Completion
        terminated = self.steps > self.total_timesteps
        truncated = terminated

        # Check for Termination
        if terminated:
            pass

        # Update Steps
        self.steps += 1

        # Information
        info = {
            "observation": self.observation,
            "accepted_sample": accepted_sample,
            "current_sample": current_sample,
            "proposed_sample": proposed_sample,
            "accepted_status": accepted_status,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }

        return self.observation, reward, terminated, truncated, info

    def reset(self, seed: Union[int, None] = None, options: Any = None):
        # Gym Recommandation
        super().reset(seed=seed)

        # Set Random Seed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # Initial Steps
        self.steps = 0

        # Initialize Observation
        self.observation = np.hstack(
            (
                np.zeros(self.sample_dim),
                self.np_random.normal(size=self.sample_dim),
            )
        )

        # Store
        self.store_observation.append(self.observation)
        self.store_log_accetance_rate.append(np.float32(0.0))
        self.store_accetped_status.append(True)
        self.store_reward.append(np.float32(0.0))

        # Information Dictionary
        info = {
            "observation": self.observation,
            "log_accetance_rate": self.store_log_accetance_rate[0],
            "accepted_status": self.store_accetped_status[0],
            "reward": self.store_reward[0],
        }

        return self.observation, info


class RLMHEnvV31(RLMHEnvBase):
    def __init__(
        self,
        log_target_pdf: Callable[[NDArray[np.float32]], np.float32],
        sample_dim: int = 2,
        total_timesteps: int = 10_000,
    ) -> None:
        super().__init__(log_target_pdf, sample_dim, total_timesteps)

        # Add Additional Store
        self._store_proposed_sample: List[NDArray[np.float32]] = []
        self._store_current_sample: List[NDArray[np.float32]] = []

        self._store_log_target_proposed: List[np.float32] = []
        self._store_log_target_current: List[np.float32] = []
        self._store_log_proposal_proposed: List[np.float32] = []
        self._store_log_proposal_current: List[np.float32] = []

        self._store_nearest_proposed_covariance: List[NDArray[np.float32]] = []
        self._store_nearest_current_covariance: List[NDArray[np.float32]] = []

    def distance_function(
        self, current_sample: NDArray[np.float32], proposed_sample: NDArray[np.float32]
    ) -> np.float32:
        return np.linalg.norm(current_sample - proposed_sample, 2)

    def reward_function(
        self,
        current_sample: NDArray[np.float32],
        proposed_sample: NDArray[np.float32],
        log_alpha: np.float32,
    ) -> np.float32:
        return np.power(
            self.distance_function(current_sample, proposed_sample), 2
        ) * np.exp(log_alpha)

    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[
        NDArray[np.float32],
        np.float32,
        bool,
        bool,
        Dict[str, Union[NDArray[np.float32], bool, np.float32]],
    ]:
        # Check Action Shape
        assert (
            action.shape[0] == 2 * self.sample_dim**2 and action.shape[0] % 2 == 0
        ), f"Action shape is {action.shape}, but expected shape is {(2 * self.sample_dim**2, )}"

        # Extract Current Sample
        current_sample, mcmc_noise = np.split(
            self.observation, [self.sample_dim], axis=0
        )

        # Extract Flat Current Covariance and Flat Proposed Covariance
        current_vector, proposed_vector = np.split(action, [self.sample_dim**2], axis=0)

        # Reshape to Covariance Matrix
        current_covariance = current_vector.reshape(self.sample_dim, self.sample_dim)
        proposed_covariance = proposed_vector.reshape(self.sample_dim, self.sample_dim)

        # Avoid Singular Covariance
        nearest_proposed_covariance: NDArray[np.float32] = Toolbox.nearestPD(proposed_covariance)
        nearest_current_covariance: NDArray[np.float32] = Toolbox.nearestPD(current_covariance)

        # Generate Proposed Sample
        proposed_sample = current_sample + np.matmul(
            mcmc_noise, np.linalg.cholesky(nearest_current_covariance)
        )

        # Calculate Log Target Density
        log_target_proposed = self.log_target_pdf(proposed_sample)
        log_target_current = self.log_target_pdf(current_sample)

        ## Avoid -np.inf
        if np.isneginf(log_target_proposed):
            log_target_proposed = -INF
        if np.isneginf(log_target_current):
            log_target_current = -INF

        # Calculate Log Proposal Densitys
        try:
            log_proposal_proposed = self.log_proposal_pdf(
                proposed_sample, current_sample, nearest_current_covariance
            )
        except Exception:
            print("nearest_current_covariance:", nearest_current_covariance)
            raise

        try:
            log_proposal_current = self.log_proposal_pdf(
                current_sample, proposed_sample, nearest_proposed_covariance
            )
        except Exception:
            print("nearest_proposed_covariance:", nearest_proposed_covariance)
            raise

        # Calculate Log Acceptance Rate
        log_alpha = np.min(
            [
                0.0,
                log_target_proposed
                - log_target_current
                + log_proposal_current
                - log_proposal_proposed,
            ]
        )

        # Accept or Reject
        if np.log(self.np_random.uniform()) < log_alpha:
            accepted_status = True
            accepted_sample = proposed_sample
        else:
            accepted_status = False
            accepted_sample = current_sample

        # Update Observation
        next_mcmc_noise = self.np_random.normal(size=self.sample_dim)
        self.observation = np.hstack((accepted_sample, next_mcmc_noise))

        # Store
        self.store_observation.append(self.observation)
        self.store_accetped_status.append(accepted_status)
        self.store_action.append(action)
        self.store_log_accetance_rate.append(log_alpha)

        self._store_proposed_sample.append(proposed_sample)
        self._store_current_sample.append(current_sample)

        self._store_log_target_proposed.append(log_target_proposed)
        self._store_log_target_current.append(log_target_current)
        self._store_log_proposal_proposed.append(log_proposal_proposed)
        self._store_log_proposal_current.append(log_proposal_current)

        self._store_nearest_proposed_covariance.append(nearest_proposed_covariance)
        self._store_nearest_current_covariance.append(nearest_current_covariance)

        # Calculate Reward
        reward = self.reward_function(current_sample, proposed_sample, log_alpha)
        self.store_reward.append(reward)

        # Check for Completion
        terminated = self.steps > self.total_timesteps
        truncated = terminated

        # Check for Termination
        if terminated:
            pass

        # Update Steps
        self.steps += 1

        # Information
        info = {
            "observation": self.observation,
            "accepted_sample": accepted_sample,
            "current_sample": current_sample,
            "proposed_sample": proposed_sample,
            "accepted_status": accepted_status,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }

        return self.observation, reward, terminated, truncated, info


class RLMHEnvV31A(RLMHEnvV31):
    def distance_function(
        self, current_sample: NDArray[np.float32], proposed_sample: NDArray[np.float32]
    ) -> np.float32:
        return np.min(np.abs(current_sample - proposed_sample))


class RLMHEnvV31B(RLMHEnvV31A):
    def reward_function(
        self,
        current_sample: NDArray[np.float32],
        proposed_sample: NDArray[np.float32],
        log_alpha: np.float32,
        weight: float = 0.7,
        kappa: float = 1.0,
    ) -> np.float32:
        return weight * expit(
            kappa
            * self.distance_function(current_sample, proposed_sample)
            * np.exp(log_alpha)
        ) + (1.0 - weight) * np.exp(self.log_target_pdf(proposed_sample))
