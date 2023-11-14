from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from gymnasium import spaces

from stable_baselines3 import TD3, DDPG
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.td3.policies import Actor

SelfDDPG = TypeVar("SelfDDPG", bound="DDPG")


class RLMCMCPolicyInterface(metaclass=ABCMeta):
    def __init__(self, sample_dim: int) -> None:
        super().__init__()
        assert isinstance(sample_dim, int) and sample_dim > 0, ValueError(
            "The sample dimension must be a positive integer."
        )
        self._sample_dim = sample_dim

    def policy_mapping(self, state: torch.Tensor) -> torch.Tensor:
        """The policy function, which is a mapping from the state to the action. Mu_{theta}: S -> A

        Args:
            state (torch.Tensor): The current state which is including the current sample and MCMC noise. s_{t} = [x_{t}, epsilon_{t}]

        Returns:
            action (torch.Tensor): The current action which is including the proposed sample at the next state and log proposal ratio at in current state. a_{t} = [x_{t+1}^{*}, L_{theta}(s_{n})]
        """
        # Extract current sample and MCMC noise
        current_sample = state[:, 0 : self._sample_dim]
        mcmc_noise = state[:, self._sample_dim :]

        # Generate proposed sample
        proposed_sample_vector = self.generate_proposed_sample(
            current_sample, mcmc_noise
        )

        # Compute log proposal ratio
        log_proposal_ratio_scalar = self.log_proposal_ratio(current_sample, mcmc_noise)

        # Construct action
        action = torch.hstack(
            [proposed_sample_vector, log_proposal_ratio_scalar.unsqueeze(1)]
        )
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

        # Extract the low rank vector and magnification
        low_rank_vector = low_rank_vector_and_magnification[:, 0 : self._sample_dim]
        magnification = low_rank_vector_and_magnification[:, -1]

        # Construct the covariance matrix
        covariance_matrix = torch.einsum(
            "ij,ik->ijk", low_rank_vector, low_rank_vector
        ) + torch.einsum("i,jk->ijk", magnification, torch.eye(self._sample_dim))

        return covariance_matrix

    @abstractmethod
    def generate_proposed_sample(
        self, current_sample: torch.Tensor, mcmc_noise: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by subclass.")

    @abstractmethod
    def log_proposal_pdf(
        self, x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by subclass.")

    def log_proposal_ratio(
        self, current_sample: torch.Tensor, mcmc_noise: torch.Tensor
    ) -> torch.Tensor:
        # Compute current covariance matrix
        current_covariance_matrix = self.parameterised_covariance_matrix(current_sample)

        # Generate proposed sample and compute proposed covariance matrix
        proposed_sample = self.generate_proposed_sample(current_sample, mcmc_noise)
        proposed_covariance_matrix = self.parameterised_covariance_matrix(
            proposed_sample
        )

        # Compute log proposal ratio
        log_proposal_pdf_current2proposed = self.log_proposal_pdf(
            current_sample, proposed_sample, proposed_covariance_matrix
        )
        log_proposal_pdf_proposed2current = self.log_proposal_pdf(
            proposed_sample, current_sample, current_covariance_matrix
        )

        # Return log proposal ratio
        return log_proposal_pdf_current2proposed - log_proposal_pdf_proposed2current


class RLMHPolicy(RLMCMCPolicyInterface):
    def generate_proposed_sample(
        self, current_sample: torch.Tensor, mcmc_noise: torch.Tensor
    ) -> torch.Tensor:
        current_covariance_matrix = self.parameterised_covariance_matrix(current_sample)
        proposed_sample = current_sample + torch.einsum(
            "ij, ijk -> ik", mcmc_noise, torch.sqrt(current_covariance_matrix)
        )

        return proposed_sample

    def log_proposal_pdf(
        self, x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor
    ) -> torch.Tensor:
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
        activation_fn: Type[nn.Module] = nn.Softplus,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_dim=features_dim,
            features_extractor=features_extractor,
            *args,
            **kwargs,
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
        activation_fn: Type[nn.Module] = nn.Softplus,
        squash_output: bool = False,
        with_bias: bool = True,
    ) -> OrderedDict[str, List[tuple[Literal[str], nn.Module]]]:
        """Create a multi layer perceptron (MLP), which is
        a collection of fully-connected layers each followed by an activation function.

        Args:
            input_dim (int): Dimension of the input vector.
            output_dim (int): Dimension of the output vector.
            net_arch (List[int]): Architecture of the neural network. It represents the number of units per layer. The length of this list is the number of layers.
            activation_fn (Type[nn.Module]): The activation function to use after each layer.
            squash_output (bool): Whether to squash the output using a Tanh activation function.
            with_bias (bool): If set to False, the layers will not learn an additive bias.

        Returns (OrderedDict[str, List[tuple[Literal[str], nn.Module]]]) aka OrderedDict[str, List[tuple[str, nn.Module]]]: The structure of the neural network that torch.Sequential can parse.

        """

        if len(net_arch) > 0:
            modules = [
                ("Input", nn.Linear(input_dim, net_arch[0], bias=with_bias)),
                ("ActivationInput", activation_fn()),
            ]
        else:
            modules = []

        for idx in range(len(net_arch) - 1):
            modules.append(
                (
                    f"Linear{idx}",
                    nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias),
                )
            )
            modules.append((f"Activation{idx}", activation_fn()))

        if output_dim > 0:
            last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
            modules.append(
                ("Output", nn.Linear(last_layer_dim, output_dim, bias=with_bias))
            )
        if squash_output:
            modules.append(("OutputActivation", nn.Tanh()))
        else:
            modules.append(("OutputActivation", activation_fn()))
        return OrderedDict(modules)

    def _actor_net(self):
        return nn.Sequential(
            self._create_mlp(
                self.sample_dim,
                self.action_dim,
                self.net_arch,
                activation_fn=nn.Softplus,
                squash_output=False,
                with_bias=True,
            )
        )

    def parameterised_low_rank_vector(self, sample: torch.Tensor) -> torch.Tensor:
        mu = self._actor_net()
        return mu(sample)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.policy_mapping(features)


class TD3Policy(BasePolicy):
    """Policy class (with both actor and critic) for TD3.

    Args:
        observation_space (spaces.Space): Observation space.
        action_space (spaces.Box): Action space.
        lr_schedule (Schedule): Learning rate schedule (could be constant).
        net_arch (Optional[Union[List[int], Dict[str, List[int]]]]): The specification of the policy and value networks.
        activation_fn (nn.Module): Activation function.
        features_extractor_class (Type[BaseFeaturesExtractor]): Features extractor to use.
        features_extractor_kwargs (Type[BaseFeaturesExtractor]): Keyword arguments. to pass to the features extractor.
        normalize_images (bool): Whether to normalize images or not, dividing by 255.0 (True by default)
        optimizer_class (Type[torch.optim.Optimizer]): The optimizer to use, ``torch.optim.Adam`` by default
        optimizer_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments, excluding the learning rate, to pass to the optimizer
        n_critics (int): Number of critic networks to create.
        share_features_extractor (bool): Whether to share or not the features extractor between the actor and the critic (this saves computation time)
    """

    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Softplus,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = [256, 256]
            else:
                net_arch = [48, 48]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(
                features_extractor=self.actor.features_extractor
            )
            self.critic_target = self.make_critic(
                features_extractor=self.actor_target.features_extractor
            )
        else:
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


class RLMHTD3Policy(TD3Policy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: List[int] | Dict[str, List[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Softplus,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )
        if net_arch is None:
            net_arch = [48, 48]

        self.net_arch = net_arch

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return RLMHPolicyActor(**actor_kwargs).to(self.device)


class RLMHTD3(TD3):
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        ):
            unscaled_action = np.array(
                [self.action_space.sample() for _ in range(n_envs)]
            )
        else:
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        if action_noise is not None:
            unscaled_action = unscaled_action + action_noise()

        buffer_action = unscaled_action
        action = buffer_action

        return action, buffer_action

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                # noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                # noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                # next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                next_actions = self.actor_target(replay_data.next_observations)

                # Compute the next Q-values: min over all critics targets
                next_q_values = torch.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )

            # Compute critic loss
            critic_loss = sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, torch.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations, self.actor(replay_data.observations)
                ).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(
                    self.actor.parameters(), self.actor_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(
                    self.critic_batch_norm_stats,
                    self.critic_batch_norm_stats_target,
                    1.0,
                )
                polyak_update(
                    self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0
                )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))


class RLMHDDPG(DDPG):
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        ):
            unscaled_action = np.array(
                [self.action_space.sample() for _ in range(n_envs)]
            )
        else:
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        if action_noise is not None:
            unscaled_action = unscaled_action + action_noise()

        buffer_action = unscaled_action
        action = buffer_action

        return action, buffer_action

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                # noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                # noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                # next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                next_actions = self.actor_target(replay_data.next_observations)

                # Compute the next Q-values: min over all critics targets
                next_q_values = torch.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )

            # Compute critic loss
            critic_loss = sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, torch.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations, self.actor(replay_data.observations)
                ).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(
                    self.actor.parameters(), self.actor_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(
                    self.critic_batch_norm_stats,
                    self.critic_batch_norm_stats_target,
                    1.0,
                )
                polyak_update(
                    self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0
                )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))


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
