import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

import torch
from torch import nn

import numpy as np

from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.td3.policies import Actor
from stable_baselines3 import TD3

SelfDDPG = TypeVar("SelfDDPG", bound="DDPG")


logging.basicConfig(level=logging.ERROR)


class RLMCMCPolicyInterface(metaclass=ABCMeta):
    def __init__(self, sample_dim: int) -> None:
        super().__init__()
        assert type(sample_dim) == int and sample_dim > 0, ValueError(
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
        logging.debug(f"covariance_matrix: {covariance_matrix}")
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

        logging.debug(f"proposed_sample_RLMHPolicy: {proposed_sample}")
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
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert (
                train_freq.unit == TrainFrequencyUnit.STEP
            ), "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if (
            action_noise is not None
            and env.num_envs > 1
            and not isinstance(action_noise, VectorizedActionNoise)
        ):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and num_collected_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(
                learning_starts, action_noise, env.num_envs
            )

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(
                replay_buffer, buffer_actions, new_obs, rewards, dones, infos
            )

            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if (
                        log_interval is not None
                        and self._episode_num % log_interval == 0
                    ):
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(
            num_collected_steps * env.num_envs,
            num_collected_episodes,
            continue_training,
        )

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        buffer_action = unscaled_action
        action = buffer_action

        return action, buffer_action


class RLMHDDPG(RLMHTD3):
    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            # Remove all tricks from TD3 to obtain DDPG:
            # we still need to specify target_policy_noise > 0 to avoid errors
            policy_delay=1,
            target_noise_clip=0.0,
            target_policy_noise=0.1,
            _init_setup_model=False,
        )

        # Use only one critic
        if "n_critics" not in self.policy_kwargs:
            self.policy_kwargs["n_critics"] = 1

        if _init_setup_model:
            self._setup_model()

    def learn(
        self: SelfDDPG,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DDPG",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDDPG:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


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


if __name__ == "__main__":
    import random
    import numpy as np

    # dim = random.randint(1, 100)
    dim = 2
    # print("dim:", dim)

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, int(2 * dim)))
    action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, dim + 1))
    features_extractor = FlattenExtractor(observation_space)

    mcmc_noise = torch.distributions.MultivariateNormal(
        torch.zeros(dim), torch.eye(dim)
    ).sample()
    state = torch.hstack(
        [
            torch.repeat_interleave(torch.tensor([0.0]), dim).unsqueeze(0),
            mcmc_noise.unsqueeze(0),
        ]
    )

    vec_state = torch.vstack([state, state, state, state, state])
    # print("vec_state:", vec_state)

    RLMHPolicyActorpolicy = RLMHPolicyActor(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=[48, 48],
        features_extractor=features_extractor,
        features_dim=int(2 * dim),
        activation_fn=nn.Softplus,
    )
    # print("RLMHPolicyActorpolicy action:", RLMHPolicyActorpolicy(state))

    TD3policy = RLMHTD3Policy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=linear_schedule(0.001),
        net_arch=[48, 48],
    )
    # print("TD3policy action:", TD3policy(state))

    print("TD3policy vec action:", TD3policy(vec_state))
