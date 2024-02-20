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

INF = np.float64(3.4028235e38)  # Corresponds to the value of FLT_MAX in C++


class RLMHEnvBase(gym.Env, ABC):
    def __init__(
        self,
        log_target_pdf: Callable[[NDArray[np.float64]], np.float64],
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
            low=-np.inf, high=np.inf, shape=(2 * sample_dim,), dtype=np.float64
        )

        # Action specification
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 * sample_dim**2,), dtype=np.float64
        )

        # Store
        self.store_observation: List[NDArray[np.float64]] = []
        self.store_action: List[NDArray[np.float64]] = []
        self.store_log_accetance_rate: List[np.float64] = []
        self.store_accetped_status: List[bool] = []
        self.store_reward: List[np.float64] = []

    def log_proposal_pdf(
        self,
        x: NDArray[np.float64],
        mean: NDArray[np.float64],
        cov: NDArray[np.float64],
    ) -> np.float64:
        return multivariate_normal.logpdf(x, mean.flatten(), cov)

    @abstractmethod
    def distance_function(
        self, current_sample: NDArray[np.float64], proposed_sample: NDArray[np.float64]
    ) -> np.float64:
        raise NotImplementedError("distance_function is not implemented.")

    @abstractmethod
    def reward_function(
        self,
        current_sample: NDArray[np.float64],
        proposed_sample: NDArray[np.float64],
        log_alpha: np.float64,
    ) -> np.float64:
        raise NotImplementedError("reward_function is not implemented.")

    def step(self, action: NDArray[np.float64]) -> Tuple[
        NDArray[np.float64],
        np.float64,
        bool,
        bool,
        Dict[str, Union[NDArray[np.float64], bool, np.float64]],
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
        nearest_proposed_covariance: NDArray[np.float64] = cov_nearest(
            proposed_covariance
        )
        nearest_current_covariance: NDArray[np.float64] = cov_nearest(
            current_covariance
        )

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

        # Initialize Store
        self.store_observation: List[NDArray[np.float64]] = []
        self.store_action: List[NDArray[np.float64]] = []
        self.store_log_accetance_rate: List[np.float64] = []
        self.store_accetped_status: List[bool] = []
        self.store_reward: List[np.float64] = []

        self.store_observation.append(self.observation)

        # Information Dictionary
        info = {
            "observation": self.observation,
            "log_accetance_rate": None,
            "accepted_status": None,
            "reward": None,
        }

        return self.observation, info


class RLMHEnvV31(RLMHEnvBase):
    def __init__(
        self,
        log_target_pdf: Callable[[NDArray[np.float64]], np.float64],
        sample_dim: int = 2,
        total_timesteps: int = 10_000,
    ) -> None:
        super().__init__(log_target_pdf, sample_dim, total_timesteps)

        # Add Additional Store
        self._store_proposed_sample: List[NDArray[np.float64]] = []
        self._store_current_sample: List[NDArray[np.float64]] = []

        self._store_log_target_proposed: List[np.float64] = []
        self._store_log_target_current: List[np.float64] = []
        self._store_log_proposal_proposed: List[np.float64] = []
        self._store_log_proposal_current: List[np.float64] = []

        self._store_nearest_proposed_covariance: List[NDArray[np.float64]] = []
        self._store_nearest_current_covariance: List[NDArray[np.float64]] = []

    def distance_function(
        self, current_sample: NDArray[np.float64], proposed_sample: NDArray[np.float64]
    ) -> np.float64:
        return np.linalg.norm(current_sample - proposed_sample, 2)

    def reward_function(
        self,
        current_sample: NDArray[np.float64],
        proposed_sample: NDArray[np.float64],
        log_alpha: np.float64,
    ) -> np.float64:
        return np.power(
            self.distance_function(current_sample, proposed_sample), 2
        ) * np.exp(log_alpha)

    def step(self, action: NDArray[np.float64]) -> Tuple[
        NDArray[np.float64],
        np.float64,
        bool,
        bool,
        Dict[str, Union[NDArray[np.float64], bool, np.float64]],
    ]:
        # Check Action Shape
        assert (
            action.shape[0] == self.sample_dim**2 << 1 and action.shape[0] % 2 == 0
        ), f"Action shape is {action.shape}, but expected shape is {(self.sample_dim**2 << 1, )}"

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
        nearest_proposed_covariance: NDArray[np.float64] = Toolbox.nearestPD(
            proposed_covariance
        )
        nearest_current_covariance: NDArray[np.float64] = Toolbox.nearestPD(
            current_covariance
        )

        # Generate Proposed Sample
        proposed_sample = current_sample + np.matmul(
            mcmc_noise, np.linalg.cholesky(nearest_current_covariance)
        )

        # Calculate Log Target Density
        log_target_proposed = self.log_target_pdf(proposed_sample)
        log_target_current = self.log_target_pdf(current_sample)

        ## Avoid -np.inf
        if np.isneginf(log_target_proposed) and np.isneginf(log_target_current):
            log_target_probability = -INF
        else:
            if np.isneginf(log_target_proposed):
                log_target_proposed = -INF
            if np.isneginf(log_target_current):
                log_target_current = -INF

            log_target_probability = log_target_proposed - log_target_current

        # Calculate Log Proposal Densitys
        log_proposal_proposed = self.log_proposal_pdf(
            proposed_sample, current_sample, nearest_current_covariance
        )
        log_proposal_current = self.log_proposal_pdf(
            current_sample, proposed_sample, nearest_proposed_covariance
        )
        log_proposal_probability = log_proposal_current - log_proposal_proposed

        # Calculate Log Acceptance Rate
        log_alpha = np.min(
            [np.float64(0.0), log_target_probability + log_proposal_probability]
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

        # Initialize Store
        self.store_observation: List[NDArray[np.float64]] = []
        self.store_action: List[NDArray[np.float64]] = []
        self.store_log_accetance_rate: List[np.float64] = []
        self.store_accetped_status: List[bool] = []
        self.store_reward: List[np.float64] = []

        self._store_proposed_sample: List[NDArray[np.float64]] = []
        self._store_current_sample: List[NDArray[np.float64]] = []

        self._store_log_target_proposed: List[np.float64] = []
        self._store_log_target_current: List[np.float64] = []
        self._store_log_proposal_proposed: List[np.float64] = []
        self._store_log_proposal_current: List[np.float64] = []

        self._store_nearest_proposed_covariance: List[NDArray[np.float64]] = []
        self._store_nearest_current_covariance: List[NDArray[np.float64]] = []

        self.store_observation.append(self.observation)

        # Information Dictionary
        info = {
            "observation": self.observation,
            "log_accetance_rate": None,
            "accepted_status": None,
            "reward": None,
        }

        return self.observation, info


class RLMHEnvV31A(RLMHEnvV31):
    def distance_function(
        self, current_sample: NDArray[np.float64], proposed_sample: NDArray[np.float64]
    ) -> np.float64:
        return np.min(np.abs(current_sample - proposed_sample))


class RLMHEnvV31B(RLMHEnvV31A):
    def reward_function(
        self,
        current_sample: NDArray[np.float64],
        proposed_sample: NDArray[np.float64],
        log_alpha: np.float64,
        weight: float = 0.7,
        kappa: float = 1.0,
    ) -> np.float64:
        return weight * expit(
            kappa
            * self.distance_function(current_sample, proposed_sample)
            * np.exp(log_alpha)
        ) + (1.0 - weight) * np.exp(self.log_target_pdf(proposed_sample))


class RLMHEnvV33(RLMHEnvV31):
    def reward_function(
        self, current_sample: NDArray[np.float64], accepted_sample: NDArray[np.float64]
    ) -> np.float64:
        return np.power(self.distance_function(current_sample, accepted_sample), 2)

    def step(self, action: NDArray[np.float64]) -> Tuple[
        NDArray[np.float64],
        np.float64,
        bool,
        bool,
        Dict[str, Union[NDArray[np.float64], bool, np.float64]],
    ]:
        # Check Action Shape
        assert (
            action.shape[0] == self.sample_dim**2 << 1 and action.shape[0] % 2 == 0
        ), f"Action shape is {action.shape}, but expected shape is {(self.sample_dim**2 << 1, )}"

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
        nearest_current_covariance: NDArray[np.float64] = Toolbox.nearestPD(
            current_covariance
        )
        nearest_proposed_covariance: NDArray[np.float64] = Toolbox.nearestPD(
            proposed_covariance
        )

        # Generate Proposed Sample
        proposed_sample = current_sample + np.matmul(
            mcmc_noise, np.linalg.cholesky(nearest_current_covariance)
        )

        # Calculate Log Target Density
        log_target_proposed = self.log_target_pdf(proposed_sample)
        log_target_current = self.log_target_pdf(current_sample)

        ## Avoid -np.inf
        if np.isneginf(log_target_proposed) and np.isneginf(log_target_current):
            log_target_probability = -INF
        else:
            if np.isneginf(log_target_proposed):
                log_target_proposed = -INF
            if np.isneginf(log_target_current):
                log_target_current = -INF

            log_target_probability = log_target_proposed - log_target_current

        # Calculate Log Proposal Densitys
        log_proposal_proposed = self.log_proposal_pdf(
            proposed_sample, current_sample, nearest_current_covariance
        )
        log_proposal_current = self.log_proposal_pdf(
            current_sample, proposed_sample, nearest_proposed_covariance
        )
        log_proposal_probability = log_proposal_current - log_proposal_proposed

        # Calculate Log Acceptance Rate
        log_alpha = np.min(
            [np.float64(0.0), log_target_probability + log_proposal_probability]
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
        reward = self.reward_function(current_sample, accepted_sample)
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


class RLMHEnvV6(RLMHEnvBase):
    def __init__(
        self,
        log_target_pdf: Callable[[NDArray[np.float64]], np.float64],
        sample_dim: int = 2,
        total_timesteps: int = 10_000,
    ) -> None:
        super().__init__(log_target_pdf, sample_dim, total_timesteps)

        # Observation specification
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(sample_dim << 1,), dtype=np.float64
        )

        # Action specification
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=((sample_dim + 1) << 1,), dtype=np.float64
        )

        # Define Observation and Current Covariance
        self.observation: NDArray[np.float64] = np.zeros(self.sample_dim)
        self.current_covariance: NDArray[np.float64] = np.eye(self.sample_dim)

        # Add Additional Store
        self._store_accepted_covariance: List[NDArray[np.float64]] = []
        self._store_proposed_sample: List[NDArray[np.float64]] = []
        self._store_current_sample: List[NDArray[np.float64]] = []

        self._store_log_target_proposed: List[np.float64] = []
        self._store_log_target_current: List[np.float64] = []
        self._store_log_proposal_proposed: List[np.float64] = []
        self._store_log_proposal_current: List[np.float64] = []

        self._store_nearest_proposed_covariance: List[NDArray[np.float64]] = []
        self._store_nearest_current_covariance: List[NDArray[np.float64]] = []

    def distance_function(
        self, current_sample: NDArray[np.float64], proposed_sample: NDArray[np.float64]
    ) -> np.float64:
        return np.linalg.norm(current_sample - proposed_sample, 2)

    def reward_function(
        self,
        current_sample: NDArray[np.float64],
        proposed_sample: NDArray[np.float64],
        log_alpha: np.float64,
    ) -> np.float64:
        return np.power(
            self.distance_function(current_sample, proposed_sample), 2
        ) * np.exp(log_alpha)

    def step(self, action: NDArray[np.float64]) -> Tuple[
        NDArray[np.float64],
        np.float64,
        bool,
        bool,
        Dict[str, Union[NDArray[np.float64], bool, np.float64]],
    ]:
        # Check Action Shape
        assert (
            action.shape[0] == (self.sample_dim + 1) << 1
        ), f"Action shape is {action.shape}, but expected shape is {((self.sample_dim + 1) << 1, )}"

        # Extract Current Sample
        current_sample, mcmc_noise = np.split(
            self.observation, [self.sample_dim], axis=0
        )

        # Extract the Current Vector and Proposed Vector
        current_vector, proposed_vector = np.split(
            action, [self.sample_dim + 1], axis=0
        )

        # Extract the Sample Vector and Covariance Magnification
        current_position, current_covariance_magnification = np.split(
            current_vector, [self.sample_dim], axis=0
        )

        proposed_position, proposed_covariance_magnification = np.split(
            proposed_vector, [self.sample_dim], axis=0
        )

        # Generate Proposed Sample
        proposed_sample = (
            current_sample
            + current_position * current_sample
            + current_covariance_magnification * mcmc_noise
        )

        # Reshape to Covariance Matrix
        current_covariance = current_covariance_magnification**2 * np.eye(
            self.sample_dim
        )
        proposed_covariance = proposed_covariance_magnification**2 * np.eye(
            self.sample_dim
        )

        # Avoid Singular Covariance
        nearest_proposed_covariance: NDArray[np.float64] = Toolbox.nearestPD(
            proposed_covariance
        )
        nearest_current_covariance: NDArray[np.float64] = Toolbox.nearestPD(
            current_covariance
        )

        # Calculate Log Target Density
        log_target_proposed = self.log_target_pdf(proposed_sample)
        log_target_current = self.log_target_pdf(current_sample)

        ## Avoid -np.inf
        if np.isneginf(log_target_proposed) and np.isneginf(log_target_current):
            log_target_probability = -INF
        else:
            if np.isneginf(log_target_proposed):
                log_target_proposed = -INF
            if np.isneginf(log_target_current):
                log_target_current = -INF

            log_target_probability = log_target_proposed - log_target_current

        # Calculate Log Proposal Densitys
        log_proposal_proposed = self.log_proposal_pdf(
            proposed_sample, current_sample, nearest_current_covariance
        )
        log_proposal_current = self.log_proposal_pdf(
            current_sample, proposed_sample, nearest_proposed_covariance
        )

        log_proposal_probability = log_proposal_current - log_proposal_proposed

        # Calculate Log Acceptance Rate
        log_alpha = np.min(
            [np.float64(0.0), log_target_probability + log_proposal_probability]
        )

        # Accept or Reject
        if np.log(self.np_random.uniform()) < log_alpha:
            accepted_status = True
            accepted_sample = proposed_sample
            accepted_covariance = proposed_covariance
        else:
            accepted_status = False
            accepted_sample = current_sample
            accepted_covariance = current_covariance

        # Update Observation
        next_mcmc_noise = self.np_random.normal(size=self.sample_dim)
        self.observation = np.hstack((accepted_sample, next_mcmc_noise))
        self.current_covariance = accepted_covariance
        self._store_accepted_covariance.append(accepted_covariance)

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

        # Initialize Current Covariance
        self.current_covariance = np.eye(self.sample_dim)

        # Initialize Store
        self.store_observation: List[NDArray[np.float64]] = []
        self.store_action: List[NDArray[np.float64]] = []
        self.store_log_accetance_rate: List[np.float64] = []
        self.store_accetped_status: List[bool] = []
        self.store_reward: List[np.float64] = []

        self._store_proposed_sample: List[NDArray[np.float64]] = []
        self._store_current_sample: List[NDArray[np.float64]] = []

        self._store_log_target_proposed: List[np.float64] = []
        self._store_log_target_current: List[np.float64] = []
        self._store_log_proposal_proposed: List[np.float64] = []
        self._store_log_proposal_current: List[np.float64] = []

        self._store_nearest_proposed_covariance: List[NDArray[np.float64]] = []
        self._store_nearest_current_covariance: List[NDArray[np.float64]] = []

        self._store_accepted_covariance: List[NDArray[np.float64]] = []

        self.store_observation.append(self.observation)
        self._store_accepted_covariance.append(self.current_covariance)

        # Information Dictionary
        info = {
            "observation": self.observation,
            "log_accetance_rate": None,
            "accepted_status": None,
            "reward": None,
        }

        return self.observation, info


class RLMHEnvV61(RLMHEnvV6):
    def reward_function(
        self,
        current_sample: NDArray[np.float64],
        accepted_sample: NDArray[np.float64],
        log_alpha: np.float64,
    ) -> np.float64:
        return np.power(self.distance_function(current_sample, accepted_sample), 2)


class RLMHEnvV7(RLMHEnvBase):
    def __init__(
        self,
        log_target_pdf: Callable[[NDArray[np.float64]], np.float64],
        sample_dim: int = 2,
        total_timesteps: int = 10_000,
    ) -> None:
        super().__init__(log_target_pdf, sample_dim, total_timesteps)

        # Observation specification
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(sample_dim,), dtype=np.float64
        )

        # Action specification
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(sample_dim + 1,), dtype=np.float64
        )

        # Define Observation and Current Covariance
        self.observation: NDArray[np.float64] = np.zeros(self.sample_dim)
        self.current_covariance: NDArray[np.float64] = np.eye(self.sample_dim)

        # Add Additional Store
        self.store_sample: List[NDArray[np.float64]] = []
        self._store_accepted_covariance: List[NDArray[np.float64]] = []
        self._store_proposed_sample: List[NDArray[np.float64]] = []
        self._store_current_sample: List[NDArray[np.float64]] = []

        self._store_log_target_proposed: List[np.float64] = []
        self._store_log_target_current: List[np.float64] = []
        self._store_log_proposal_proposed: List[np.float64] = []
        self._store_log_proposal_current: List[np.float64] = []

        self._store_nearest_proposed_covariance: List[NDArray[np.float64]] = []
        self._store_nearest_current_covariance: List[NDArray[np.float64]] = []

    def distance_function(
        self, current_sample: NDArray[np.float64], proposed_sample: NDArray[np.float64]
    ) -> np.float64:
        return np.linalg.norm(current_sample - proposed_sample, 2)

    def reward_function(
        self,
        current_sample: NDArray[np.float64],
        proposed_sample: NDArray[np.float64],
        log_alpha: np.float64,
    ) -> np.float64:
        return np.power(
            self.distance_function(current_sample, proposed_sample), 2
        ) * np.exp(log_alpha)

    def step(self, action: NDArray[np.float64]) -> Tuple[
        NDArray[np.float64],
        np.float64,
        bool,
        bool,
        Dict[str, Union[NDArray[np.float64], bool, np.float64]],
    ]:
        # Check Action Shape
        assert (
            action.shape[0] == self.sample_dim + 1
        ), f"Action shape is {action.shape}, but expected shape is {(self.sample_dim + 1, )}"

        # Assign Observation to Current Sample
        current_sample = self.accepted_sample

        # Extract Proposed low rank vector and Magnification
        proposed_low_rank_vector, proposed_magnification = np.split(
            action, [self.sample_dim], axis=0
        )

        # Restore to Covariance Matrix
        current_covariance = self.current_covariance
        proposed_covariance = np.outer(
            proposed_low_rank_vector, proposed_low_rank_vector
        ) + proposed_magnification**2 * np.eye(self.sample_dim)

        # Avoid Singular Covariance
        nearest_proposed_covariance: NDArray[np.float64] = Toolbox.nearestPD(
            proposed_covariance
        )
        nearest_current_covariance: NDArray[np.float64] = Toolbox.nearestPD(
            current_covariance
        )

        # Generate Proposed Sample
        proposed_sample = self.observation

        # Calculate Log Target Density
        log_target_proposed = self.log_target_pdf(proposed_sample)
        log_target_current = self.log_target_pdf(current_sample)

        ## Avoid -np.inf
        if np.isneginf(log_target_proposed) and np.isneginf(log_target_current):
            log_target_probability = -INF
        else:
            if np.isneginf(log_target_proposed):
                log_target_proposed = -INF
            if np.isneginf(log_target_current):
                log_target_current = -INF

            log_target_probability = log_target_proposed - log_target_current

        # Calculate Log Proposal Densitys
        log_proposal_proposed = self.log_proposal_pdf(
            proposed_sample, current_sample, nearest_current_covariance
        )
        log_proposal_current = self.log_proposal_pdf(
            current_sample, proposed_sample, nearest_proposed_covariance
        )

        log_proposal_probability = log_proposal_current - log_proposal_proposed

        # Calculate Log Acceptance Rate
        log_alpha = np.min(
            [np.float64(0.0), log_target_probability + log_proposal_probability]
        )

        # Accept or Reject
        if np.log(self.np_random.uniform()) < log_alpha:
            accepted_status = True
            accepted_sample = proposed_sample
            accepted_covariance = proposed_covariance
        else:
            accepted_status = False
            accepted_sample = current_sample
            accepted_covariance = current_covariance

        # Update Observation
        self.accepted_sample = accepted_sample
        self.observation = multivariate_normal.rvs(
            mean=accepted_sample, cov=accepted_covariance, size=1
        ).reshape(-1)
        self._store_accepted_covariance.append(accepted_covariance)
        self.current_covariance = accepted_covariance

        # Store
        self.store_sample.append(accepted_sample)
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

    def reset(self, seed: Union[int, None] = None, options: Any = None):
        # Gym Recommandation
        super().reset(seed=seed)

        # Set Random Seed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # Initial Steps
        self.steps = 0

        # Initialize Accepted Sample
        self.accepted_sample = np.zeros(self.sample_dim)

        # Initialize Observation
        self.observation = multivariate_normal.rvs(
            mean=self.accepted_sample, cov=np.eye(self.sample_dim), size=1
        ).reshape(-1)

        # Initialize Current Covariance
        self.store_observation: List[NDArray[np.float64]] = []
        self.store_action: List[NDArray[np.float64]] = []
        self.store_log_accetance_rate: List[np.float64] = []
        self.store_accetped_status: List[bool] = []
        self.store_reward: List[np.float64] = []

        self._store_proposed_sample: List[NDArray[np.float64]] = []
        self._store_current_sample: List[NDArray[np.float64]] = []

        self._store_log_target_proposed: List[np.float64] = []
        self._store_log_target_current: List[np.float64] = []
        self._store_log_proposal_proposed: List[np.float64] = []
        self._store_log_proposal_current: List[np.float64] = []

        self._store_nearest_proposed_covariance: List[NDArray[np.float64]] = []
        self._store_nearest_current_covariance: List[NDArray[np.float64]] = []

        self._store_accepted_covariance: List[NDArray[np.float64]] = []

        self.store_observation.append(self.observation)
        self.store_sample.append(self.observation)
        self._store_accepted_covariance.append(self.current_covariance)

        # Information Dictionary
        info = {
            "observation": self.observation,
            "accepted_sample": self.accepted_sample,
            "log_accetance_rate": None,
            "accepted_status": None,
            "reward": None,
        }

        return self.observation, info


class RLMHEnvV71(RLMHEnvV7):
    def reward_function(
        self,
        current_sample: NDArray[np.float64],
        accepted_sample: NDArray[np.float64],
        log_alpha: np.float64,
    ) -> np.float64:
        return np.power(self.distance_function(current_sample, accepted_sample), 2)


class RLMHEnvV8(RLMHEnvBase):
    def __init__(
        self,
        log_target_pdf: Callable[[NDArray[np.float64]], np.float64],
        sample_dim: int = 2,
        total_timesteps: int = 10_000,
    ) -> None:
        super().__init__(log_target_pdf, sample_dim, total_timesteps)

        # Observation specification
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(sample_dim << 1,), dtype=np.float64
        )

        # Action specification
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=((sample_dim + 1) << 1,), dtype=np.float64
        )

        # Define Observation and Current Covariance
        self.observation: NDArray[np.float64] = np.zeros(self.sample_dim)
        self.current_covariance: NDArray[np.float64] = np.eye(self.sample_dim)

        # Add Additional Store
        self._store_accepted_covariance: List[NDArray[np.float64]] = []
        self._store_proposed_sample: List[NDArray[np.float64]] = []
        self._store_current_sample: List[NDArray[np.float64]] = []

        self._store_log_target_proposed: List[np.float64] = []
        self._store_log_target_current: List[np.float64] = []
        self._store_log_proposal_proposed: List[np.float64] = []
        self._store_log_proposal_current: List[np.float64] = []

        self._store_nearest_proposed_covariance: List[NDArray[np.float64]] = []
        self._store_nearest_current_covariance: List[NDArray[np.float64]] = []

    def distance_function(
        self, current_sample: NDArray[np.float64], proposed_sample: NDArray[np.float64]
    ) -> np.float64:
        return np.linalg.norm(current_sample - proposed_sample, 2)

    def reward_function(
        self,
        current_sample: NDArray[np.float64],
        proposed_sample: NDArray[np.float64],
        log_alpha: np.float64,
    ) -> np.float64:
        return np.power(
            self.distance_function(current_sample, proposed_sample), 2
        ) * np.exp(log_alpha)

    def step(self, action: NDArray[np.float64]) -> Tuple[
        NDArray[np.float64],
        np.float64,
        bool,
        bool,
        Dict[str, Union[NDArray[np.float64], bool, np.float64]],
    ]:
        # Check Action Shape
        assert (
            action.shape[0] == (self.sample_dim + 1) << 1
        ), f"Action shape is {action.shape}, but expected shape is {((self.sample_dim + 1) << 1, )}"

        # Extract Current and Proposed Sample
        current_sample, proposed_sample = np.split(
            self.observation, [self.sample_dim], axis=0
        )

        # Extract the Current Vector and Proposed Vector
        (
            current_low_rank_vector_and_magnification,
            proposed_low_rank_vector_and_magnification,
        ) = np.split(action, [self.sample_dim + 1], axis=0)

        # Extract the Sample Vector and Covariance Magnification
        current_vector, current_covariance_magnification = np.split(
            current_low_rank_vector_and_magnification, [self.sample_dim], axis=0
        )
        proposed_vector, proposed_covariance_magnification = np.split(
            proposed_low_rank_vector_and_magnification, [self.sample_dim], axis=0
        )

        # Restore Covariance Matrix
        current_covariance = current_covariance_magnification**2 * np.eye(
            self.sample_dim
        )
        proposed_covariance = proposed_covariance_magnification**2 * np.eye(
            self.sample_dim
        )

        # Avoid Singular Covariance
        nearest_proposed_covariance: NDArray[np.float64] = Toolbox.nearestPD(
            proposed_covariance
        )
        nearest_current_covariance: NDArray[np.float64] = Toolbox.nearestPD(
            current_covariance
        )

        # Calculate Log Target Density
        log_target_proposed = self.log_target_pdf(proposed_sample)
        log_target_current = self.log_target_pdf(current_sample)

        ## Avoid -np.inf
        if np.isneginf(log_target_proposed) and np.isneginf(log_target_current):
            log_target_probability = -INF
        else:
            if np.isneginf(log_target_proposed):
                log_target_proposed = -INF
            if np.isneginf(log_target_current):
                log_target_current = -INF

            log_target_probability = log_target_proposed - log_target_current

        # Calculate Log Proposal Densitys
        log_proposal_proposed = self.log_proposal_pdf(
            proposed_sample, current_sample, nearest_current_covariance
        )
        log_proposal_current = self.log_proposal_pdf(
            current_sample, proposed_sample, nearest_proposed_covariance
        )

        log_proposal_probability = log_proposal_current - log_proposal_proposed

        # Calculate Log Acceptance Rate
        log_alpha = np.min(
            [np.float64(0.0), log_target_probability + log_proposal_probability]
        )

        # Accept or Reject
        if np.log(self.np_random.uniform()) < log_alpha:
            accepted_status = True
            accepted_sample = proposed_sample
            accepted_vector = proposed_vector
            accepted_covariance = proposed_covariance
        else:
            accepted_status = False
            accepted_sample = current_sample
            accepted_vector = current_vector
            accepted_covariance = current_covariance

        # Update Observation
        next_proposed_sample = multivariate_normal.rvs(
            mean=(accepted_sample + accepted_vector * accepted_sample), cov=accepted_covariance, size=1
        ).reshape(-1)
        self.observation = np.hstack((accepted_sample, next_proposed_sample))
        self.current_covariance = accepted_covariance
        self._store_accepted_covariance.append(accepted_covariance)

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

        # Initialize Current Covariance
        self.current_covariance = np.eye(self.sample_dim)

        # Initialize Store
        self.store_observation: List[NDArray[np.float64]] = []
        self.store_action: List[NDArray[np.float64]] = []
        self.store_log_accetance_rate: List[np.float64] = []
        self.store_accetped_status: List[bool] = []
        self.store_reward: List[np.float64] = []

        self._store_proposed_sample: List[NDArray[np.float64]] = []
        self._store_current_sample: List[NDArray[np.float64]] = []

        self._store_log_target_proposed: List[np.float64] = []
        self._store_log_target_current: List[np.float64] = []
        self._store_log_proposal_proposed: List[np.float64] = []
        self._store_log_proposal_current: List[np.float64] = []

        self._store_nearest_proposed_covariance: List[NDArray[np.float64]] = []
        self._store_nearest_current_covariance: List[NDArray[np.float64]] = []

        self._store_accepted_covariance: List[NDArray[np.float64]] = []

        self.store_observation.append(self.observation)
        self._store_accepted_covariance.append(self.current_covariance)

        # Information Dictionary
        info = {
            "observation": self.observation,
            "log_accetance_rate": None,
            "accepted_status": None,
            "reward": None,
        }

        return self.observation, info


class RLMHEnvV81(RLMHEnvV8):
    def reward_function(
        self,
        current_sample: NDArray[np.float64],
        accepted_sample: NDArray[np.float64],
        log_alpha: np.float64,
    ) -> np.float64:
        return np.power(self.distance_function(current_sample, accepted_sample), 2)


class RLMHEnvV82(RLMHEnvV8):
    def __init__(
        self,
        log_target_pdf: Callable[[NDArray[np.float64]], np.float64],
        sample_dim: int = 2,
        total_timesteps: int = 10_000,
    ) -> None:
        super().__init__(log_target_pdf, sample_dim, total_timesteps)
        # Action specification
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=((sample_dim) << 1,), dtype=np.float64
        )

    def step(self, action: NDArray[np.float64]) -> Tuple[
        NDArray[np.float64],
        np.float64,
        bool,
        bool,
        Dict[str, Union[NDArray[np.float64], bool, np.float64]],
    ]:
        # Check Action Shape
        assert (
            action.shape[0] == (self.sample_dim) << 1
        ), f"Action shape is {action.shape}, but expected shape is {((self.sample_dim) << 1, )}"

        # Extract Current and Proposed Sample
        current_sample, proposed_sample = np.split(
            self.observation, [self.sample_dim], axis=0
        )

        # Extract the Current Vector and Proposed Vector
        (
            current_vector,
            proposed_vector,
        ) = np.split(action, [self.sample_dim], axis=0)

        # Calculate Log Target Density
        log_target_proposed = self.log_target_pdf(proposed_sample)
        log_target_current = self.log_target_pdf(current_sample)

        ## Avoid -np.inf
        if np.isneginf(log_target_proposed) and np.isneginf(log_target_current):
            log_target_probability = -INF
        else:
            if np.isneginf(log_target_proposed):
                log_target_proposed = -INF
            if np.isneginf(log_target_current):
                log_target_current = -INF

            log_target_probability = log_target_proposed - log_target_current

        # Calculate Log Proposal Densitys
        log_proposal_proposed = self.log_proposal_pdf(
            proposed_sample, current_sample, np.eye(self.sample_dim)
        )
        log_proposal_current = self.log_proposal_pdf(
            current_sample, proposed_sample, np.eye(self.sample_dim)
        )

        log_proposal_probability = log_proposal_current - log_proposal_proposed

        # Calculate Log Acceptance Rate
        log_alpha = np.min(
            [np.float64(0.0), log_target_probability + log_proposal_probability]
        )

        # Accept or Reject
        if np.log(self.np_random.uniform()) < log_alpha:
            accepted_status = True
            accepted_sample = proposed_sample
            accepted_vector = proposed_vector
        else:
            accepted_status = False
            accepted_sample = current_sample
            accepted_vector = current_vector

        # Update Observation
        next_proposed_sample = multivariate_normal.rvs(
            mean=(accepted_sample + accepted_vector * accepted_sample), cov=np.eye(self.sample_dim), size=1
        ).reshape(-1)
        self.observation = np.hstack((accepted_sample, next_proposed_sample))

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

        # Initialize Current Covariance
        self.current_covariance = np.eye(self.sample_dim)

        # Initialize Store
        self.store_observation: List[NDArray[np.float64]] = []
        self.store_action: List[NDArray[np.float64]] = []
        self.store_log_accetance_rate: List[np.float64] = []
        self.store_accetped_status: List[bool] = []
        self.store_reward: List[np.float64] = []

        self._store_proposed_sample: List[NDArray[np.float64]] = []
        self._store_current_sample: List[NDArray[np.float64]] = []

        self._store_log_target_proposed: List[np.float64] = []
        self._store_log_target_current: List[np.float64] = []
        self._store_log_proposal_proposed: List[np.float64] = []
        self._store_log_proposal_current: List[np.float64] = []

        self._store_nearest_proposed_covariance: List[NDArray[np.float64]] = []
        self._store_nearest_current_covariance: List[NDArray[np.float64]] = []

        self._store_accepted_covariance: List[NDArray[np.float64]] = []

        self.store_observation.append(self.observation)
        self._store_accepted_covariance.append(self.current_covariance)

        # Information Dictionary
        info = {
            "observation": self.observation,
            "log_accetance_rate": None,
            "accepted_status": None,
            "reward": None,
        }

        return self.observation, info


class RLMHEnvV83(RLMHEnvV82):
    def reward_function(
        self,
        current_sample: NDArray[np.float64],
        proposed_sample: NDArray[np.float64],
        log_alpha: np.float64,
    ) -> np.float64:
        reward = np.power(
            self.distance_function(current_sample, proposed_sample), 2
        ) * np.exp(log_alpha)
        return 1 / (1 + np.exp(-reward))


class RLMHEnvV84(RLMHEnvV82):
    def reward_function(
        self,
        current_sample: NDArray[np.float64],
        proposed_sample: NDArray[np.float64],
        log_alpha: np.float64,
    ) -> np.float64:
        reward = np.power(
            self.distance_function(current_sample, proposed_sample), 2
        )
        return 1 / (1 + np.exp(-reward))

    def step(self, action: NDArray[np.float64]) -> Tuple[
        NDArray[np.float64],
        np.float64,
        bool,
        bool,
        Dict[str, Union[NDArray[np.float64], bool, np.float64]],
    ]:
        # Check Action Shape
        assert (
            action.shape[0] == (self.sample_dim) << 1
        ), f"Action shape is {action.shape}, but expected shape is {((self.sample_dim) << 1, )}"

        # Extract Current and Proposed Sample
        current_sample, proposed_sample = np.split(
            self.observation, [self.sample_dim], axis=0
        )

        # Extract the Current Vector and Proposed Vector
        (
            current_vector,
            proposed_vector,
        ) = np.split(action, [self.sample_dim], axis=0)

        # Calculate Log Target Density
        log_target_proposed = self.log_target_pdf(proposed_sample)
        log_target_current = self.log_target_pdf(current_sample)

        ## Avoid -np.inf
        if np.isneginf(log_target_proposed) and np.isneginf(log_target_current):
            log_target_probability = -INF
        else:
            if np.isneginf(log_target_proposed):
                log_target_proposed = -INF
            if np.isneginf(log_target_current):
                log_target_current = -INF

            log_target_probability = log_target_proposed - log_target_current

        # Calculate Log Proposal Densitys
        log_proposal_proposed = self.log_proposal_pdf(
            proposed_sample, current_sample, np.eye(self.sample_dim)
        )
        log_proposal_current = self.log_proposal_pdf(
            current_sample, proposed_sample, np.eye(self.sample_dim)
        )

        log_proposal_probability = log_proposal_current - log_proposal_proposed

        # Calculate Log Acceptance Rate
        log_alpha = np.min(
            [np.float64(0.0), log_target_probability + log_proposal_probability]
        )

        # Accept or Reject
        if np.log(self.np_random.uniform()) < log_alpha:
            accepted_status = True
            accepted_sample = proposed_sample
            accepted_vector = proposed_vector
        else:
            accepted_status = False
            accepted_sample = current_sample
            accepted_vector = current_vector

        # Update Observation
        next_proposed_sample = multivariate_normal.rvs(
            mean=(accepted_sample + accepted_vector * accepted_sample), cov=np.eye(self.sample_dim), size=1
        ).reshape(-1)
        self.observation = np.hstack((accepted_sample, next_proposed_sample))

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

        # Calculate Reward
        reward = self.reward_function(current_sample, accepted_sample, log_alpha)
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
