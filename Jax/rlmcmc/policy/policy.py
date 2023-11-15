from abc import ABCMeta, abstractmethod
from typing import Callable, List
import gymnasium as gym

import jax
from jax import numpy as jnp
from jax.typing import ArrayLike

import flax
from flax import linen as nn
from flax.training.train_state import TrainState

import optax

key = jax.random.PRNGKey(1234)
key, actor_key, critic_key = jax.random.split(key, 3)


class RLMCMCPolicyInterface(metaclass=ABCMeta):
    def __init__(self, sample_dim: int) -> None:
        super().__init__()
        assert isinstance(sample_dim, int) and sample_dim > 0, ValueError(
            "The sample dimension must be a positive integer."
        )
        self._sample_dim = sample_dim

    def policy_mapping(self, state: jnp.ndarray) -> jnp.ndarray:
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
        action = jnp.hstack(
            [proposed_sample_vector, jnp.expand_dims(log_proposal_ratio_scalar, axis=1)]
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
    def parameterised_low_rank_vector(self, sample: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Must be implemented by subclass.")

    def parameterised_covariance_matrix(self, sample: jnp.ndarray) -> jnp.ndarray:
        low_rank_vector_and_magnification = self.parameterised_low_rank_vector(sample)

        # Extract the low rank vector and magnification
        low_rank_vector = low_rank_vector_and_magnification[:, 0 : self._sample_dim]
        magnification = low_rank_vector_and_magnification[:, -1]

        # Construct the covariance matrix
        covariance_matrix = jnp.einsum(
            "ij,ik->ijk", low_rank_vector, low_rank_vector
        ) + jnp.einsum("i,jk->ijk", magnification, jnp.eye(self._sample_dim))

        return covariance_matrix

    @abstractmethod
    def generate_proposed_sample(
        self, current_sample: jnp.ndarray, mcmc_noise: jnp.ndarray
    ) -> jnp.ndarray:
        raise NotImplementedError("Must be implemented by subclass.")

    @abstractmethod
    def log_proposal_pdf(
        self, x: jnp.ndarray, mean: jnp.ndarray, cov: jnp.ndarray
    ) -> jnp.ndarray:
        raise NotImplementedError("Must be implemented by subclass.")

    def log_proposal_ratio(
        self, current_sample: jnp.ndarray, mcmc_noise: jnp.ndarray
    ) -> jnp.ndarray:
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
        self, current_sample: jnp.ndarray, mcmc_noise: jnp.ndarray
    ) -> jnp.ndarray:
        current_covariance_matrix = self.parameterised_covariance_matrix(current_sample)

        proposed_sample = current_sample + jnp.einsum(
            "ij, ijk -> ik", mcmc_noise, jnp.sqrt(current_covariance_matrix)
        )

        return proposed_sample

    def log_proposal_pdf(
        self, x: jnp.ndarray, mean: jnp.ndarray, cov: jnp.ndarray
    ) -> ArrayLike:
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean, cov)


class RLMHPolicyActor(RLMHPolicy):
    def __init__(
        self, state, learning_rate: float = 0.001, actor_key: dict = actor_key
    ):
        super().__init__(sample_dim=int(0.5 * state.shape[1]))
        self.sample = state[0:1, 0 : self.sample_dim]
        self.learning_rate = learning_rate
        self.actor_key = actor_key

        self.init_actor_state()

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        return self.policy_mapping(obs)

    class Actor(nn.Module):
        """
        Actor Network
        """

        output_dim: int
        net_arch: List[int]
        activation_fn: Callable[
            [jax.typing.ArrayLike], jax.typing.ArrayLike
        ] = nn.softplus
        squash_output: bool = False
        with_bias: bool = True

        @nn.compact
        def __call__(self, x):
            return nn.Sequential(self._create_mlp())(x)

        def _create_mlp(self) -> List:
            """Create a multi layer perceptron (MLP), which is
            a collection of fully-connected layers each followed by an activation function.

            Args:
                input_dim (int): Dimension of the input vector.
                output_dim (int): Dimension of the output vector.
                net_arch (List[int]): Architecture of the neural network. It represents the number of units per layer. The length of this list is the number of layers.
                activation_fn (Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike]): The activation function to use after each layer.
                squash_output (bool): Whether to squash the output using a Tanh activation function.
                with_bias (bool): If set to False, the layers will not learn an additive bias.

            Returns (List: The structure of the neural network that nn.Sequential can parse.

            """

            # Initialize the neural network
            if not self.net_arch:
                self.net_arch = [64, 64]

            # Create the neural network
            modules = []

            for idx in range(len(self.net_arch)):
                modules.append(
                    nn.Dense(
                        self.net_arch[idx], use_bias=self.with_bias, name=f"Dense{idx}"
                    )
                )
                modules.append(self.activation_fn)

            if self.output_dim > 0:
                modules.append(
                    nn.Dense(self.output_dim, use_bias=self.with_bias, name="Output")
                )
            if self.squash_output:
                modules.append(nn.tanh)
            else:
                modules.append(self.activation_fn)

            return modules

    class TrainState(TrainState):
        target_params: flax.core.FrozenDict

    def init_actor_state(self, net_arch=[48, 48]):
        action_dim = self.sample_dim + 1
        actor = self.Actor(output_dim=action_dim, net_arch=net_arch)
        init_actor_state = self.TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(self.actor_key, self.sample),
            target_params=actor.init(self.actor_key, self.sample),
            tx=optax.adam(self.learning_rate),
        )
        self.actor_state = init_actor_state

    def actor_forward(self, x):
        return jax.jit(self.actor_state.apply_fn)(self.actor_state.params, x)

    def parameterised_low_rank_vector(self, sample: jnp.ndarray) -> jnp.ndarray:
        return self.actor_forward(sample)

    def actor_params_forward(self, params: dict, state: jnp.ndarray):
        return jax.jit(self.actor_state.apply_fn)(params, state)


class RLMHPolicyCritic:
    def __init__(
        self, env: gym.Env, learning_rate: float = 0.001, critic_key: dict = critic_key
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.critic_key = critic_key

        self.init_critic_state()

    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.critic_forward(obs, action)

    class Critic(nn.Module):
        """
        Critic Network
        """

        net_arch: List[int]
        activation_fn: Callable[
            [jax.typing.ArrayLike], jax.typing.ArrayLike
        ] = nn.softplus
        squash_output: bool = False
        with_bias: bool = True

        @nn.compact
        def __call__(self, state: jnp.ndarray, action: jnp.ndarray):
            return nn.Sequential(self._create_mlp())(
                jnp.concatenate([state, action], -1)
            )

        def _create_mlp(self) -> List:
            """Create a multi layer perceptron (MLP), which is
            a collection of fully-connected layers each followed by an activation function.

            Args:
                input_dim (int): Dimension of the input vector.
                output_dim (int): Dimension of the output vector.
                net_arch (List[int]): Architecture of the neural network. It represents the number of units per layer. The length of this list is the number of layers.
                activation_fn (Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike]): The activation function to use after each layer.
                squash_output (bool): Whether to squash the output using a Tanh activation function.
                with_bias (bool): If set to False, the layers will not learn an additive bias.

            Returns (List: The structure of the neural network that nn.Sequential can parse.

            """

            # Initialize the neural network
            if not self.net_arch:
                self.net_arch = [64, 64]

            # Create the neural network
            modules = []

            for idx in range(len(self.net_arch)):
                modules.append(
                    nn.Dense(
                        self.net_arch[idx], use_bias=self.with_bias, name=f"Dense{idx}"
                    )
                )
                modules.append(self.activation_fn)

            modules.append(nn.Dense(1, use_bias=self.with_bias, name="Output"))

            if self.squash_output:
                modules.append(nn.tanh)
            else:
                modules.append(self.activation_fn)

            return modules

    class TrainState(TrainState):
        target_params: flax.core.FrozenDict

    def init_critic_state(self, net_arch=[48, 48]):
        critic = self.Critic(net_arch=net_arch)
        init_critic_state = self.TrainState.create(
            apply_fn=critic.apply,
            params=critic.init(
                self.critic_key,
                self.env.observation_space.sample(),
                self.env.action_space.sample(),
            ),
            target_params=critic.init(
                self.critic_key,
                self.env.observation_space.sample(),
                self.env.action_space.sample(),
            ),
            tx=optax.adam(self.learning_rate),
        )
        self.critic_state = init_critic_state

    def critic_forward(self, state: jnp.ndarray, action: jnp.ndarray):
        return jax.jit(self.critic_state.apply_fn)(
            self.critic_state.params, state, action
        )

    def critic_params_forward(self, params: dict, state: jnp.ndarray, action: jnp.ndarray):
        return jax.jit(self.critic_state.apply_fn)(params, state, action)
