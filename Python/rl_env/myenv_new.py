import gymnasium as gym
from gymnasium import spaces

import random
import numpy as np
from scipy.stats import multivariate_normal

import jax
import flax
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

import toml
from typing import Sequence
from types import SimpleNamespace

INF = 3.4028235e+38 # Corresponds to the value of FLT_MAX in C++
SEED = 1234
generator = np.random.Generator(np.random.PCG64(SEED))

# seeding
random.seed(SEED)
np.random.seed(SEED)
key = jax.random.PRNGKey(SEED)
key, actor_key, qf1_key = jax.random.split(key, 3)


class Policy:
    def __init__(self, dim=2, exploration_noise=0.01, action_range=[0.0000001, INF]):
        self.dim = dim
        self.exploration_noise = exploration_noise
        self.action_range = action_range

    def __call__(self, state):
        x = state[0:self.dim]
        a1 = x + self.actor(x) @ state[self.dim:]
        a2 = self.L(state)
        return np.hstack((a1, a2))

    def actor_flatten(self, x):
        """
        Temporary replacement for neural networks to test.
        """
        # return np.hstack((np.arange(1, int(0.5 * (1+self.dim)*self.dim)), 100.0))
        raise NotImplementedError

    def actor(self, x):
        cov_flatten = (
            self.actor_flatten(x) + \
            generator.normal(0.0, self.exploration_noise)
            ).clip(
                self.action_range[0], self.action_range[1]
        )
        # cov_flatten = self.actor_flatten(x)

        tri_matrix = np.zeros((self.dim, self.dim))
        index = 0
        for i in range(self.dim):
            for j in range(self.dim):
                tri_matrix[i, j] = cov_flatten[index]
                index += 1

        cov_mat = tri_matrix @ tri_matrix.T + cov_flatten[-1] * np.eye(self.dim)

        return cov_mat

    def log_q(self, x, mean, cov):
        """
        Multivariate normal distribution.
        """
        # return (-0.5 * len(mean) * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov)) - 0.5 * (x - mean) @ np.linalg.inv(cov) @ (x - mean).T).squeeze()
        return multivariate_normal.logpdf(x, mean=mean, cov=cov)

    def L(self, state):
        x = state[0:self.dim]
        cov_x = self.actor(x)

        x_star = generator.multivariate_normal(x, cov_x)
        cov_x_star = self.actor(x_star)

        log_q_prop = self.log_q(x_star, x, cov_x)
        log_q_curr = self.log_q(x, x_star, cov_x_star)

        return log_q_curr - log_q_prop

class MyEnvNew(gym.Env):
    def __init__(self, log_p, dim=2, max_steps=10_000):
        super(MyEnvNew, self).__init__()

        # Target Distribution
        self.log_p = log_p

        # Parameter
        self.dim = dim
        self.max_steps = max_steps
        self.ts = 0  # iteration time

        # Observation specification
        self.observation_space = spaces.Box(low=-INF, high=INF, shape=(2*dim,))

        # Action specification
        self.action_space = spaces.Box(low=0.0, high=INF, shape=(int(0.5 * (1 + dim) * dim + 1),))

        # Store
        self.store_state = []
        self.store_log_accetance_rate = []
        self.store_accetped_status = []
        # self.store_action = []
        self.store_reward = []

    def step(self, action):
        # Extract x
        x = self.state[0:self.dim]
        # Extract x_star and transition kernel
        x_star, log_q_kernel = action[0:self.dim], action[self.dim]

        # Accept/Reject Process
        log_alpha = self.log_p(x_star) - self.log_p(x) + log_q_kernel

        if np.log(generator.uniform()) < log_alpha:
            accepted_status = True
            x_accept = x_star
        else:
            accepted_status = False
            x_accept = x

        state = np.hstack((x_accept, generator.normal(size=(self.dim,))))

        # Store
        self.store_state.append(state)
        self.store_accetped_status.append(accepted_status)
        # self.store_action.append(action)
        self.store_log_accetance_rate.append(log_alpha)

        # Calculate Reward
        reward = np.power(np.linalg.norm(x - x_star, 2), 2) * np.exp(log_alpha)
        self.store_reward.append(reward)

        # Update Iteration Time
        self.state = state
        self.ts += 1

        # Check for Completion
        terminated = self.ts >= self.max_steps
        truncated = terminated
        if terminated:
            # self.reset()
            pass

        # Information
        info = {
            "state": x_accept,
            "accepted_status": accepted_status,
            "reward": reward,
            "theta_prop": x_star
        }

        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.ts = 0
        self.state = np.hstack((np.zeros(self.dim), generator.normal(size=(self.dim,))))  # initialize s_{t}
        self.store_state.append(self.state)
        self.store_accetped_status.append(True)
        # self.store_action.append(np.array([1.0]))
        self.store_reward.append(0.0)
        self.store_log_accetance_rate.append(np.array([0.0]))

        # Information
        info = {
            "state": self.state,
            "accepted_status": True,
            "reward": 0.0
        }

        return self.state, info
