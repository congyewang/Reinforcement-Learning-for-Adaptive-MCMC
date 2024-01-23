from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import wishart
import seaborn as sns
from matplotlib import pyplot as plt

import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer

import time
from tqdm.auto import trange
from typing import Generic, List, TypeVar, Union
from numpy.typing import NDArray

LearningBase = TypeVar("LearningBase", bound="LearningBase")
LearningDDPG = TypeVar("LearningDDPG", bound="LearningDDPG")
LearningTD3 = TypeVar("LearningTD3", bound="LearningTD3")


class LearningBase(ABC, Generic[LearningBase]):
    def __init__(self) -> None:
        self._last_called = None

    @abstractmethod
    def train(self: LearningBase) -> LearningBase:
        raise NotImplementedError("train method is not implemented")

    @abstractmethod
    def predict(
        self: LearningBase, predicted_env: gym.spaces.Box, predicted_timesteps: int
    ) -> LearningBase:
        raise NotImplementedError("predict method is not implemented")

    @abstractmethod
    def save(self, folder_path: str) -> None:
        raise NotImplementedError("save method is not implemented")

    def plot(
        self,
        scatter: bool = True,
        trace: bool = True,
        cov_trace: bool = True,
        hist: bool = True,
        kde: bool = True,
        immediate_reward: bool = True,
        cumulative_reward: bool = True,
        cov: bool = True,
        target: bool = False,
        critic_loss: bool = False,
        actor_values: bool = False,
        *args,
        **kwargs,
    ) -> None:
        if self._last_called == self.train.__name__:
            env = self.env
        elif self._last_called == self.predict.__name__:
            env = self.predicted_env
        else:
            raise ValueError("train or predict method should be called first")

        assert env.num_envs == 1, "only single environment is supported"

        unwrapped_env = env.unwrapped.envs[0].env.env.env
        samples = np.array(unwrapped_env.store_observation)[
            :, 0 : unwrapped_env.sample_dim
        ]
        covariances = np.array(unwrapped_env.store_action)[
            :, 0 : unwrapped_env.sample_dim**2
        ]
        rewards = np.array(unwrapped_env.store_reward)
        acceptance_rate = np.sum(unwrapped_env.store_accetped_status) / len(
            unwrapped_env.store_accetped_status
        )

        unnecessary_samples = samples.shape[0] - covariances.shape[0]

        if scatter:
            if samples.shape[1] == 2:
                plt.scatter(
                    samples[unnecessary_samples:, 0],
                    samples[unnecessary_samples:, 1],
                    c=(covariances[:, 0] + covariances[:, 3]),
                    cmap="viridis",
                )
                plt.colorbar(label="Trace of the Covariance")
                plt.title("Scatter Plot")
                plt.show()

        if trace:
            for i in range(samples.shape[1]):
                plt.plot(samples[:, i], label=f"dim {i}", alpha=0.5)
            plt.legend()
            plt.title(f"Trace Plot - Acc: {acceptance_rate:.4f}")
            plt.show()

            if samples.shape[1] == 2:
                plt.plot(
                    samples[unnecessary_samples:, 0],
                    samples[unnecessary_samples:, 1],
                    "o-",
                    color="grey",
                    alpha=0.05,
                )
                plt.title("2D Trace Plot")
                plt.show()

        if cov_trace:
            if samples.shape[1] == 2:
                plt.plot(covariances[:, 0] + covariances[:, 3])
                plt.title("Trace of the Covariance")
                plt.show()

        if hist:
            for i in range(samples.shape[1]):
                plt.hist(samples[:, i], bins=30, label=f"dim {i}", alpha=0.5)
            plt.legend()
            plt.title("Histogram of the Samples")
            plt.show()

        if kde:
            for i in range(samples.shape[1]):
                sns.kdeplot(samples[:, i], label=f"dim {i}", alpha=0.5)
            plt.legend()
            plt.title("KDE of the Samples")
            plt.show()

        if immediate_reward:
            plt.plot(rewards)
            plt.title("Immediate Reward")
            plt.show()

        if cumulative_reward:
            plt.plot(np.cumsum(rewards))
            plt.title("Cumulative Reward")
            plt.show()

        if cov:
            for i in range(covariances.shape[1]):
                plt.plot(covariances[:, i], label=f"cov {i}", alpha=0.5)
            plt.legend()
            plt.title("Trace Plot of the Covariance in Each Dimension")
            plt.show()

        if target:
            _num = 1000
            _x = np.linspace(-5, 5, _num)
            _y = np.linspace(5, 20, _num)
            _X, _Y = np.meshgrid(_x, _y)

            _Z = np.zeros((_num, _num))

            for i in range(len(_x)):
                for j in range(len(_y)):
                    _Z[i, j] = np.exp(
                        unwrapped_env.log_target_pdf(np.array([_x[i], _y[j]]))
                    )

            plt.contourf(_X, _Y, _Z.T, 50, cmap="viridis")
            plt.colorbar()
            plt.title("Target Distribution")
            plt.show()

        if critic_loss:
            plt.plot(np.array(self.critic_loss))
            plt.title("Critic Loss")
            plt.show()

        if actor_values:
            plt.plot(np.array(self.actor_values))
            plt.title("Actor Values")
            plt.show()

    def dataframe(self) -> pd.DataFrame:
        if self._last_called == self.train.__name__:
            env = self.env
        elif self._last_called == self.predict.__name__:
            env = self.predicted_env
        else:
            raise ValueError("train or predict method should be called first")

        assert env.num_envs == 1, "only single environment is supported"
        unwrapped_env = env.unwrapped.envs[0].env.env.env
        assert unwrapped_env.sample_dim == 2, "only 2D sample space is supported"

        samples = np.array(unwrapped_env.store_observation)[
            :, 0 : unwrapped_env.sample_dim
        ]
        proposed_samples = np.array(unwrapped_env._store_proposed_sample)
        covariances = np.array(unwrapped_env.store_action)
        rewards = np.array(unwrapped_env.store_reward).reshape(-1, 1)
        log_accetance_rate = np.array(unwrapped_env.store_log_accetance_rate).reshape(
            -1, 1
        )
        accetped_status = np.array(unwrapped_env.store_accetped_status).reshape(-1, 1)

        df = pd.DataFrame(
            np.hstack(
                [
                    samples[:-1],
                    proposed_samples,
                    covariances,
                    rewards,
                    log_accetance_rate,
                    accetped_status,
                ]
            ),
            columns=[
                "x",
                "y",
                "proposed_x",
                "proposed_y",
                "cov1",
                "cov2",
                "cov3",
                "cov4",
                "proposed_cov1",
                "proposed_cov2",
                "proposed_cov3",
                "proposed_cov4",
                "rewards",
                "log_alpha",
                "accepted_status",
            ],
        )

        return df


class LearningDDPG(LearningBase, Generic[LearningDDPG]):
    def __init__(
        self,
        env: gym.spaces.Box,
        actor: torch.nn.Module,
        target_actor: torch.nn.Module,
        critic: torch.nn.Module,
        target_critic: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        total_timesteps: int = 10_000,
        learning_starts: int = 32,
        batch_size: int = 32,
        exploration_noise: float = 0.1,
        gamma: float = 0.99,
        policy_frequency: int = 2,
        tau: float = 0.005,
        seed: Union[int, None] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        assert isinstance(
            env.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"
        self.env = env

        self.obs, self.infos = env.reset(seed=seed)
        self.sample_dim: int = np.prod(env.single_observation_space.shape) >> 1

        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.replay_buffer = replay_buffer

        self.total_timesteps = total_timesteps
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.gamma = gamma
        self.policy_frequency = policy_frequency
        self.tau = tau
        self.seed = seed
        self.device = device

        self.critic_loss = []
        self.actor_values = []

        self.predicted_observation: List[NDArray[np.float64]] = []
        self.predicted_action: List[NDArray[np.float64]] = []
        self.predicted_reward: List[NDArray[np.float64]] = []

    def soft_clipping(self, g: torch.Tensor, t: float = 1.0, p: int = 2):
        """
        Soft clipping function for gradient clipping.
        """

        norm = torch.norm(g, p=p)
        return t / (t + norm) * g

    def train(self: LearningDDPG, gradient_clipping: bool = False) -> LearningDDPG:
        """
        Training Session for DDPG.
        """
        if gradient_clipping:
            for p in self.critic.parameters():
                p.register_hook(self.soft_clipping)

            for p in self.actor.parameters():
                p.register_hook(self.soft_clipping)

        for global_step in trange(self.total_timesteps):
            if global_step < self.learning_starts:
                actions = np.array(
                    [
                        np.hstack(
                            (
                                np.eye(self.sample_dim, dtype=np.float64).reshape(
                                    -1, self.sample_dim << 1
                                ),
                                np.eye(self.sample_dim, dtype=np.float64).reshape(
                                    -1, self.sample_dim << 1
                                ),
                            )
                        )
                        for _ in range(self.env.num_envs)
                    ]
                ).reshape(1, -1)
            else:
                with torch.no_grad():
                    actions = self.actor(torch.from_numpy(self.obs).to(self.device))

            next_obs, rewards, terminations, truncations, self.infos = self.env.step(
                actions
            )

            real_next_obs = next_obs.copy()
            self.replay_buffer.add(
                self.obs, real_next_obs, actions, rewards, terminations, self.infos
            )

            self.obs = next_obs

            if global_step > self.learning_starts:
                data = self.replay_buffer.sample(self.batch_size)
                with torch.no_grad():
                    next_state_actions = self.target_actor(data.next_observations)
                    critic_next_target = self.target_critic(
                        data.next_observations, next_state_actions
                    )
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.gamma * (critic_next_target).view(-1)
                critic_a_values = self.critic(data.observations, data.actions).view(-1)
                critic_loss = F.mse_loss(critic_a_values, next_q_value)

                # optimize the model
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                if global_step % self.policy_frequency == 0:
                    actor_loss = -self.critic(
                        data.observations, self.actor(data.observations)
                    ).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # update the target network
                    for param, target_param in zip(
                        self.actor.parameters(), self.target_actor.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.critic.parameters(), self.target_critic.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )

                if global_step % 100 == 0:
                    self.critic_loss.append(critic_loss.item())
                    self.actor_values.append(actor_loss.item())

        self._last_called = self.train.__name__
        return self

    def predict(
        self: LearningDDPG,
        predicted_env: gym.spaces.Box,
        predicted_timesteps: int = 10_000,
    ) -> LearningDDPG:
        assert isinstance(
            predicted_env.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"

        self.predicted_env = predicted_env

        # Reset the environment
        predicted_obs, _ = predicted_env.reset(seed=self.seed)

        # Store predicted obs, action, and reward
        predicted_observation: List[NDArray[np.float64]] = []
        predicted_action: List[NDArray[np.float64]] = []
        predicted_reward: List[NDArray[np.float64]] = []

        for _ in trange(predicted_timesteps):
            with torch.no_grad():
                predicted_actions = self.actor(
                    torch.from_numpy(predicted_obs).to(self.device)
                )

            predicted_obs, predicted_rewards, _, _, _ = predicted_env.step(
                predicted_actions
            )

            predicted_observation.append(predicted_obs)
            predicted_action.append(predicted_actions.view(-1).detach().cpu().numpy())
            predicted_reward.append(predicted_rewards)

        self.predicted_observation = np.array(predicted_observation).reshape(
            -1, np.prod(predicted_env.single_observation_space.shape)
        )
        self.predicted_action = np.array(predicted_action)
        self.predicted_reward = np.array(predicted_reward).flatten()

        self._last_called = self.predict.__name__
        return self

    def save(self, folder_path: str) -> None:
        model_path = f"{folder_path}/ddpg.{time.time()}.pth"
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
            model_path,
        )


class LearningTD3(LearningBase):
    pass


class LearningDDPGRandom(LearningDDPG):
    def train(self: LearningDDPG, gradient_clipping: bool = False) -> LearningDDPG:
        """
        Training Session for DDPG.
        """
        if gradient_clipping:
            for p in self.critic.parameters():
                p.register_hook(self.soft_clipping)

            for p in self.actor.parameters():
                p.register_hook(self.soft_clipping)

        for global_step in trange(self.total_timesteps):
            if global_step < self.learning_starts:
                actions = np.array(
                    [
                        wishart.rvs(self.sample_dim, np.eye(self.sample_dim), size=2).reshape(-1, self.sample_dim << 2)
                        for _ in range(self.env.num_envs)
                    ]
                ).reshape(1, -1)
            else:
                with torch.no_grad():
                    actions = self.actor(torch.from_numpy(self.obs).to(self.device))
                    actions += torch.normal(0, torch.ones_like(actions) * self.exploration_noise)
                    actions = actions.cpu().numpy().clip(self.env.single_action_space.low, self.env.single_action_space.high)

            next_obs, rewards, terminations, truncations, self.infos = self.env.step(
                actions
            )

            real_next_obs = next_obs.copy()
            self.replay_buffer.add(
                self.obs, real_next_obs, actions, rewards, terminations, self.infos
            )

            self.obs = next_obs

            if global_step > self.learning_starts:
                data = self.replay_buffer.sample(self.batch_size)
                with torch.no_grad():
                    next_state_actions = self.target_actor(data.next_observations)
                    critic_next_target = self.target_critic(
                        data.next_observations, next_state_actions
                    )
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.gamma * (critic_next_target).view(-1)
                critic_a_values = self.critic(data.observations, data.actions).view(-1)
                critic_loss = F.mse_loss(critic_a_values, next_q_value)

                # optimize the model
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                if global_step % self.policy_frequency == 0:
                    actor_loss = -self.critic(
                        data.observations, self.actor(data.observations)
                    ).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # update the target network
                    for param, target_param in zip(
                        self.actor.parameters(), self.target_actor.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.critic.parameters(), self.target_critic.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )

                if global_step % 100 == 0:
                    self.critic_loss.append(critic_loss.item())
                    self.actor_values.append(actor_loss.item())

        self._last_called = self.train.__name__
        return self