import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gymnasium as gym


class Actor(nn.Module):
    def __init__(self, env: gym.spaces.Box, sample_dim: int = 2) -> None:
        super().__init__()

        self.sample_dim = sample_dim

        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc_mu = nn.Linear(24, np.prod(env.single_action_space.shape))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        "Mu function"
        current_sample, mcmc_noise = torch.split(observation, self.sample_dim, dim=1)
        proposed_sample = self.generate_proposed_sample(current_sample, mcmc_noise)

        current_covariance_flatten = self.covariance(current_sample).reshape(
            -1, 2 * self.sample_dim
        )
        proposed_covariance_flatten = self.covariance(proposed_sample).reshape(
            -1, 2 * self.sample_dim
        )

        return torch.hstack([current_covariance_flatten, proposed_covariance_flatten])

    def low_rank_vector_and_magnification(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_mu(x)
        return x

    def covariance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Restored low rank vector and magnification to covariance matrix.
        """
        low_rank_vector, mag = torch.split(
            self.low_rank_vector_and_magnification(x), self.sample_dim, dim=1
        )

        return (
            low_rank_vector[:, :, None] * low_rank_vector[:, None, :]
            + mag[:, :, None] ** 2 * torch.eye(self.sample_dim)[None, :, :]
        )

    def generate_proposed_sample(
        self, current_sample: torch.Tensor, mcmc_noise: torch.Tensor
    ) -> torch.Tensor:
        return current_sample + torch.einsum(
            "ij,ijk->ik", mcmc_noise, torch.linalg.cholesky(self.covariance(current_sample))
        )
