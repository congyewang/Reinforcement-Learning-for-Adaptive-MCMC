import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gymnasium as gym


class Actor(nn.Module):
    def __init__(
        self, env: gym.spaces.Box, device: torch.device = torch.device("cpu")
    ) -> None:
        super().__init__()
        self.device = device

        self.sample_dim = int(np.array(env.single_observation_space.shape).prod()) >> 1


        self.fc1 = nn.Linear(self.sample_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc_mu = nn.Linear(8, self.sample_dim + 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        "Mu function"
        current_sample, mcmc_noise = torch.split(
            observation, [self.sample_dim, self.sample_dim], dim=1
        )
        proposed_sample = self.generate_proposed_sample(current_sample, mcmc_noise)

        current_covariance_flatten = self.covariance(current_sample).reshape(
            -1, 2 * self.sample_dim
        )
        proposed_covariance_flatten = self.covariance(proposed_sample).reshape(
            -1, 2 * self.sample_dim
        )

        return torch.hstack([current_covariance_flatten, proposed_covariance_flatten])

    def low_rank_vector_and_magnification(self, x: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc_mu(x)
        return x

    def covariance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Restored low rank vector and magnification to covariance matrix.
        """
        low_rank_vector, mag = torch.split(
            self.low_rank_vector_and_magnification(x), [self.sample_dim, 1], dim=1
        )

        return (
            low_rank_vector[:, :, None] * low_rank_vector[:, None, :]
            + mag[:, :, None] ** 2
            * torch.eye(self.sample_dim).to(self.device)[None, :, :]
        )

    def generate_proposed_sample(
        self, current_sample: torch.Tensor, mcmc_noise: torch.Tensor
    ) -> torch.Tensor:
        current_covariance = self.covariance(current_sample)
        try:
            L: torch.Tensor = torch.linalg.cholesky(current_covariance)
        except Exception:
            print("current_covariance:", current_covariance)
            raise
        return current_sample + torch.einsum(
            "ij,ijk->ik",
            mcmc_noise,
            L,
        )
