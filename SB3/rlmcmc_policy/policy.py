from abc import ABCMeta, abstractmethod
from ast import Tuple
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    Callable
    )
import torch
from torch import nn
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import DDPG
from stable_baselines3.td3.policies import TD3Policy
from  stable_baselines3.td3.policies import Actor
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    FlattenExtractor,
    BaseFeaturesExtractor,
    NatureCNN,
    get_actor_critic_arch
    )
from stable_baselines3.common.type_aliases import Schedule
from collections import OrderedDict


class RLMCMCPolicyInterface(metaclass=ABCMeta): 
    def __init__(self, sample_dim: int) -> None:
        super().__init__()
        assert type(sample_dim) == int and sample_dim > 0, ValueError("The sample dimension must be a positive integer.")
        self._sample_dim = sample_dim

    def policy_mapping(self, state: torch.Tensor) -> torch.Tensor:
        """The policy function, which is a mapping from the state to the action. Mu_{theta}: S -> A

        Args:
            state (torch.Tensor): The current state which is including the current sample and MCMC noise. s_{t} = [x_{t}, epsilon_{t}]

        Returns:
            action (torch.Tensor): The current action which is including the proposed sample at the next state and log proposal ratio at in current state. a_{t} = [x_{t+1}^{*}, L_{theta}(s_{n})]
        """
        # Extract current sample and MCMC noise
        current_sample = state[:, 0:self._sample_dim]
        mcmc_noise = state[:, self._sample_dim:]

        # Generate proposed sample
        proposed_sample_vector = self.generate_proposed_sample(current_sample, mcmc_noise)

        # Compute log proposal ratio
        log_proposal_ratio_scalar = self.log_proposal_ratio(current_sample, mcmc_noise)

        # Construct action
        action = torch.hstack([proposed_sample_vector, log_proposal_ratio_scalar.unsqueeze(0)])
        return action

    @property
    def sample_dim(self) -> int:
        """The dimension of the sample space.
        Args:
            void
        Returns:
            _sample_dim (int): The dimension of the single sample.
        """
        return self._sample_dim

    @abstractmethod
    def parameterised_low_rank_vector(self, sample: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by subclass.")

    def parameterised_covariance_matrix(self, sample: torch.Tensor) -> torch.Tensor:
        low_rank_vector_and_magnification = self.parameterised_low_rank_vector(sample)

        #Extract the low rank vector and magnification
        low_rank_vector = low_rank_vector_and_magnification[:, 0:self._sample_dim]
        magnification = low_rank_vector_and_magnification[:, -1]

        #Construct the covariance matrix
        covariance_matrix = low_rank_vector @ low_rank_vector.T + magnification * torch.eye(self._sample_dim)
        print("covariance_matrix:", covariance_matrix)
        return covariance_matrix

    @abstractmethod
    def generate_proposed_sample(self, current_sample: torch.Tensor, mcmc_noise: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by subclass.")

    @abstractmethod
    def log_proposal_pdf(self, x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by subclass.")

    def log_proposal_ratio(self, current_sample: torch.Tensor, mcmc_noise: torch.Tensor) -> torch.Tensor:
        # Compute current covariance matrix
        current_covariance_matrix = self.parameterised_covariance_matrix(current_sample)

        # Generate proposed sample and compute proposed covariance matrix
        proposed_sample = self.generate_proposed_sample(current_sample, mcmc_noise)
        proposed_covariance_matrix = self.parameterised_covariance_matrix(proposed_sample)

        # Compute log proposal ratio
        log_proposal_pdf_current2proposed = self.log_proposal_pdf(current_sample, proposed_sample, proposed_covariance_matrix)
        log_proposal_pdf_proposed2current = self.log_proposal_pdf(proposed_sample, current_sample, current_covariance_matrix)

        # Return log proposal ratio
        return log_proposal_pdf_current2proposed - log_proposal_pdf_proposed2current

class RLMHPolicy(RLMCMCPolicyInterface):
    def generate_proposed_sample(self, current_sample: torch.Tensor, mcmc_noise: torch.Tensor) -> torch.Tensor:
        current_covariance_matrix = self.parameterised_covariance_matrix(current_sample) 
        return current_sample + mcmc_noise @ torch.sqrt(current_covariance_matrix)

    def log_proposal_pdf(self, x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        multivariate_normal = torch.distributions.MultivariateNormal(mean, cov)
        return multivariate_normal.log_prob(x)

class RLMHPolicyActor(Actor, RLMHPolicy):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_dim=features_dim,
            features_extractor=features_extractor,
            *args,
            **kwargs
        )

        super(RLMHPolicy, self).__init__(sample_dim=int(0.5 * features_dim))

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        self.action_dim = get_action_dim(self.action_space)

    def _create_mlp(
            self,
            input_dim: int,
            output_dim: int,
            net_arch: List[int],
            activation_fn: Type[nn.Module] = nn.ReLU,
            squash_output: bool = False,
            with_bias: bool = True,
    ) -> OrderedDict[str, List[tuple[Literal[str], nn.Module]]]:
        """
        Create a multi layer perceptron (MLP), which is
        a collection of fully-connected layers each followed by an activation function.

        :param input_dim: Dimension of the input vector
        :param output_dim:
        :param net_arch: Architecture of the neural net
            It represents the number of units per layer.
            The length of this list is the number of layers.
        :param activation_fn: The activation function
            to use after each layer.
        :param squash_output: Whether to squash the output using a Tanh
            activation function
        :param with_bias: If set to False, the layers will not learn an additive bias
        :return:
        """

        if len(net_arch) > 0:
            modules = [
                ('Input', nn.Linear(input_dim, net_arch[0], bias=with_bias)),
                ('ActivationInput', activation_fn())
                ]
        else:
            modules = []

        for idx in range(len(net_arch) - 1):
            modules.append((f'Linear{idx}', nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias)))
            modules.append((f'Activation{idx}', activation_fn()))

        if output_dim > 0:
            last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
            modules.append(
                ('Output', nn.Linear(last_layer_dim, output_dim, bias=with_bias))
                )
        if squash_output:
            modules.append(
                ('OutputActivation', nn.Tanh())
                )
        else:
            modules.append(
                ('OutputActivation', activation_fn())
                )
        return OrderedDict(modules)

    def _actor_net(self):
        return nn.Sequential(
                self._create_mlp(
                    self.sample_dim,
                    self.action_dim,
                    self.net_arch,
                    activation_fn=nn.Softplus,
                    squash_output=False,
                    with_bias=True
                )
        )

    def parameterised_low_rank_vector(self, sample: torch.Tensor) -> torch.Tensor:
        mu = self._actor_net()
        return mu(sample)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.policy_mapping(features)

class RLMHTD3Policy(TD3Policy):
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return RLMHPolicyActor(**actor_kwargs).to(self.device)

def linear_schedule(initial_value: float = 0.001) -> Schedule:
    """Linear learning rate schedule.

    Args:
        initial_value (torch.Tensor): Initial learning rate.

    Returns (Callable[[float], float]) aka Schedule in sb3):
        schedule that computes current learning rate depending on remaining progress.
    """
    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0.

        Args:
            Progress_remaining (float)
        Returns (float):
            Current learning rate depending on remaining progress.
        """
        return progress_remaining * initial_value

    return func
