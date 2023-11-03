import numpy as np
from abc import ABCMeta, abstractmethod
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class RLMCMCPolicyInterface(metaclass=ABCMeta):
    def __init__(self, dim: int) -> None:
        if type(dim) != int:
            raise TypeError("dim can only accept int")

        self.dim = dim

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        # Extract current sample and MCMC noise
        current_sample = state[0:self.dim]
        mcmc_noise = state[self.dim:]

        # Generate proposed sample
        proposed_sample_vector = self.generate_proposed_sample(current_sample, mcmc_noise)

        # Compute log proposal ratio
        log_proposal_ratio_scalar = self.log_proposal_ratio(current_sample, mcmc_noise)

        # Construct action
        action = torch.hstack([proposed_sample_vector, log_proposal_ratio_scalar])
        return action

    @abstractmethod
    def parameterised_low_rank_vector(self, sample: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by subclass")

    def parameterised_covariance_matrix(self, sample: torch.Tensor) -> torch.Tensor:
        low_rank_vector_and_magnification = self.parameterised_low_rank_vector(sample)

        #Extract the low rank vector and magnification
        low_rank_vector = low_rank_vector_and_magnification[0:self.dim]
        magnification = low_rank_vector_and_magnification[-1]

        #Construct the covariance matrix
        low_rank_matrix = low_rank_vector.unsqueeze(1)
        covariance_matrix = low_rank_matrix @ low_rank_matrix.T + magnification * torch.eye(self.dim)

        return covariance_matrix

    @abstractmethod
    def generate_proposed_sample(self, current_sample: torch.Tensor, mcmc_noise: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by subclass")

    @abstractmethod
    def log_proposal_pdf(self, x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by subclass")

    def log_proposal_ratio(self, current_sample: torch.Tensor, mcmc_noise: torch.Tensor) -> torch.Tensor:
        current_covariance_matrix = self.parameterised_covariance_matrix(current_sample)

        proposed_sample = self.generate_proposed_sample(current_sample, mcmc_noise)
        proposed_covariance_matrix = self.parameterised_covariance_matrix(proposed_sample)

        log_proposal_pdf_current2proposed = self.log_proposal_pdf(current_sample, proposed_sample, proposed_covariance_matrix)
        log_proposal_pdf_proposed2current = self.log_proposal_pdf(proposed_sample, current_sample, current_covariance_matrix)

        return log_proposal_pdf_current2proposed - log_proposal_pdf_proposed2current

class RLMHPolicy(RLMCMCPolicyInterface):
    def __init__(self, dim=2):
        super().__init__(dim)

    def parameterised_low_rank_vector(self, sample: torch.Tensor) -> torch.Tensor:
        return torch.Tensor([1., 0., 0., 1.])

    def generate_proposed_sample(self, current_sample: torch.Tensor, mcmc_noise: torch.Tensor) -> torch.Tensor:
        current_covariance_matrix = self.parameterised_covariance_matrix(current_sample) 
        return current_sample + mcmc_noise @ torch.sqrt(current_covariance_matrix)

    def log_proposal_pdf(self, x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        multivariate_normal = torch.distributions.MultivariateNormal(mean, cov)
        return multivariate_normal.log_prob(x)

class RLMHCustomActorCritics(BaseFeaturesExtractor):
    pass


if __name__ == "__main__":
    policy = RLMHPolicy(dim=3)
    mcmc_noise = torch.distributions.MultivariateNormal(torch.zeros(3), torch.eye(3)).sample()
    state = torch.hstack([torch.Tensor([0., 0., 0.]), mcmc_noise])
    action = policy(state)

    print(action)
