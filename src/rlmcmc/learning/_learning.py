from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer

import time
from tqdm.auto import trange
from typing import Any, Union, List
from numpy.typing import NDArray


class PredictedPlot:
    def __init__(self, pointer):
        self.pointer = pointer

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
        *args,
        **kwargs,
    ) -> None:
        assert (
            self.pointer.predicted_env.num_envs == 1
        ), "only single environment is supported"

        unwrapped_env = self.pointer.predicted_env.unwrapped.envs[0].env.env.env
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


class LearningBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError("train method is not implemented")

    @abstractmethod
    def predict(self, predicted_env: gym.spaces.Box, predicted_timesteps: int) -> Union[Any, None]:
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
        *args,
        **kwargs,
    ) -> None:
        assert self.env.num_envs == 1, "only single environment is supported"

        unwrapped_env = self.env.unwrapped.envs[0].env.env.env
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


class LearningDDPG(LearningBase):
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
        self.gamma = gamma
        self.policy_frequency = policy_frequency
        self.tau = tau
        self.seed = seed
        self.device = device

        self.predicted_observation: List[NDArray[np.float64]] = []
        self.predicted_action: List[NDArray[np.float64]] = []
        self.predicted_reward: List[NDArray[np.float64]] = []

    def train(self) -> None:
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

    def predict(
        self, predicted_env: gym.spaces.Box, predicted_timesteps: int = 10_000
    ) -> PredictedPlot:
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

        return PredictedPlot(self)

    def save(self, folder_path: str) -> None:
        model_path = f"{folder_path}/ddpg.{time.time()}.pth"
        torch.save((self.actor.state_dict(), self.critic.state_dict()), model_path)

    def dataframe(self) -> pd.DataFrame:
        assert self.env.num_envs == 1, "only single environment is supported"
        unwrapped_env = self.env.unwrapped.envs[0].env.env.env
        assert unwrapped_env.sample_dim == 2, "only 2D sample space is supported"

        samples = np.array(unwrapped_env.store_observation)[
            :, 0 : unwrapped_env.sample_dim
        ]
        covariances = np.array(unwrapped_env.store_action)[
            :, 0 : unwrapped_env.sample_dim**2
        ]
        rewards = np.array(unwrapped_env.store_reward)
        log_accetance_rate = np.array(unwrapped_env.store_log_accetance_rate).reshape(
            -1, 1
        )
        accetped_status = np.array(unwrapped_env.store_accetped_status).reshape(-1, 1)

        unnecessary_samples = samples.shape[0] - covariances.shape[0]
        df = pd.DataFrame(
            np.hstack(
                [
                    samples[unnecessary_samples:],
                    covariances,
                    rewards.reshape(-1, 1)[unnecessary_samples:],
                    log_accetance_rate[unnecessary_samples:],
                    accetped_status[unnecessary_samples:],
                ]
            ),
            columns=[
                "x",
                "y",
                "cov1",
                "cov2",
                "cov3",
                "cov4",
                "rewards",
                "log_alpha",
                "accepted_status",
            ],
        )

        return df


class LearningTD3(LearningBase):
    pass
