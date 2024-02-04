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
from torch.optim.optimizer import Optimizer as Optimizer
from tqdm.auto import trange
from typing import Generic, List, TypeVar, Union
from numpy.typing import NDArray

from ..utils import Toolbox

import wandb

LearningInterface = TypeVar("LearningInterface", bound="LearningInterface")
LearningDDPG = TypeVar("LearningDDPG", bound="LearningDDPG")
LearningTD3 = TypeVar("LearningTD3", bound="LearningTD3")


class LearningInterface(ABC, Generic[LearningInterface]):
    def __init__(self) -> None:
        self._last_called = None

    @abstractmethod
    def train(self: LearningInterface) -> LearningInterface:
        raise NotImplementedError("train method is not implemented")

    def predict(
        self,
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
                predicted_actions.detach().cpu().numpy()
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

    @abstractmethod
    def save(self, folder_path: str) -> None:
        raise NotImplementedError("save method is not implemented")

    def soft_clipping(self, g: torch.Tensor, t: float = 1.0, p: int = 2):
        """
        Soft clipping function for gradient clipping.
        """

        norm = torch.norm(g, p=p)
        return t / (t + norm) * g

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
        critic_loss: bool = False,
        actor_loss: bool = False,
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
                wandb.log({f"Scatter Plot {self._last_called}": wandb.Image(plt)})
                plt.clf()

        if trace:
            for i in range(samples.shape[1]):
                data_trace = [[j, x] for (j, x) in enumerate(samples[:, i])]
                table_trace = wandb.Table(data=data_trace, columns=["step", "x"])
                wandb.log(
                    {
                        f"trace_plot_dim{i} {self._last_called}": wandb.plot.line(
                            table_trace,
                            "step",
                            "x",
                            title=f"Trace Plot at Dim {i} ({self._last_called})",
                        )
                    }
                )
            if samples.shape[1] == 2:
                plt.plot(
                    samples[unnecessary_samples:, 0],
                    samples[unnecessary_samples:, 1],
                    "o-",
                )
                plt.xlabel("x")
                plt.ylabel("y")
                plt.title(f"2D Trace Plot ({self._last_called})")
                wandb.log({f"2D Trace Plot {self._last_called}": plt})
                plt.clf()

        if cov_trace:
            data_cov_trace = np.sum(
                np.diagonal(
                    covariances.reshape(-1, self.sample_dim, self.sample_dim),
                    axis1=1,
                    axis2=2,
                ),
                axis=1,
            )
            data_idx_cov_trace = [[j, x] for (j, x) in enumerate(data_cov_trace)]
            table_cov_trace = wandb.Table(
                data=data_idx_cov_trace, columns=["step", "trace"]
            )
            wandb.log(
                {
                    f"cov_trace {self._last_called}": wandb.plot.line(
                        table_cov_trace,
                        "step",
                        "trace",
                        title=f"Trace of Covariance Plot ({self._last_called})",
                    )
                }
            )

        if hist:
            for i in range(samples.shape[1]):
                data_hist = samples[:, i].reshape(-1, 1)
                table_hist = wandb.Table(data=data_hist, columns=["samples"])
                wandb.log(
                    {
                        f"hist dim{i} {self._last_called}": wandb.plot.histogram(
                            table_hist,
                            "samples",
                            title=f"Histogram Plot at Dim {i} ({self._last_called})",
                        )
                    }
                )

        if kde:
            for i in range(samples.shape[1]):
                sns.kdeplot(samples[:, i], label=f"dim {i}", alpha=0.5)
            plt.legend()
            plt.title("KDE of the Samples")
            wandb.log({f"KDE Plot {self._last_called}": wandb.Image(plt)})
            plt.clf()

        if immediate_reward:
            data_immediate_rewards = [[i, x] for (i, x) in enumerate(rewards)]
            table_immediate_rewards = wandb.Table(
                data=data_immediate_rewards, columns=["step", "rewards"]
            )
            wandb.log(
                {
                    f"immediate_reward {self._last_called}": wandb.plot.line(
                        table_immediate_rewards,
                        "step",
                        "rewards",
                        title=f"Immediate Reward ({self._last_called})",
                    )
                }
            )

        if cumulative_reward:
            data_cumulative_rewards = [
                [i, x] for (i, x) in enumerate(np.cumsum(rewards))
            ]
            table_cumulative_rewards = wandb.Table(
                data=data_cumulative_rewards, columns=["step", "rewards"]
            )
            wandb.log(
                {
                    f"cumulative_reward {self._last_called}": wandb.plot.line(
                        table_cumulative_rewards,
                        "step",
                        "rewards",
                        title=f"Cumulative Reward ({self._last_called})",
                    )
                }
            )

        if cov:
            for i in range(covariances.shape[1]):
                plt.plot(covariances[:, i], label=f"cov {i}", alpha=0.5)
            plt.legend()
            plt.title("Trace Plot of the Covariance in Each Dimension")
            wandb.log({f"cov {self._last_called}": plt})
            plt.clf()

        if critic_loss:
            data_critic_loss = [
                [i * 100, x] for (i, x) in enumerate(np.array(self.critic_loss))
            ]
            table_critic_loss = wandb.Table(
                data=data_critic_loss, columns=["step", "critic_loss"]
            )
            wandb.log(
                {
                    f"ritic_loss {self._last_called}": wandb.plot.line(
                        table_critic_loss,
                        "step",
                        "critic_loss",
                        title=f"Critic Loss ({self._last_called})",
                    )
                }
            )

        if actor_loss:
            data_actor_loss = [
                [i * 100, x] for (i, x) in enumerate(np.array(self.actor_loss))
            ]
            table_actor_loss = wandb.Table(
                data=data_actor_loss, columns=["step", "actor_loss"]
            )
            wandb.log(
                {
                    f"ritic_loss {self._last_called}": wandb.plot.line(
                        table_actor_loss,
                        "step",
                        "actor_loss",
                        title=f"Actor Loss ({self._last_called})",
                    )
                }
            )

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


class LearningDDPG(LearningInterface, Generic[LearningDDPG]):
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
        device: torch.device = torch.device("cpu"),
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

        self.critic_loss: List[float] = []
        self.actor_loss: List[float] = []

        self.predicted_observation: List[NDArray[np.float64]] = []
        self.predicted_action: List[NDArray[np.float64]] = []
        self.predicted_reward: List[NDArray[np.float64]] = []

    def train(self: LearningDDPG, gradient_clipping: bool = False) -> LearningDDPG:
        """
        Training Session for DDPG.
        """
        if gradient_clipping:
            for p_critic in self.critic.parameters():
                p_critic.register_hook(self.soft_clipping)

            for p_actor in self.actor.parameters():
                p_actor.register_hook(self.soft_clipping)

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
                    actions += torch.normal(
                        0, torch.ones_like(actions) * self.exploration_noise
                    )
                    actions = (
                        actions.cpu()
                        .numpy()
                        .clip(
                            self.env.single_action_space.low,
                            self.env.single_action_space.high,
                        )
                    )
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
                    self.actor_loss.append(actor_loss.item())

        self._last_called = self.train.__name__
        return self

    def save(self, folder_path: str) -> None:
        model_path = f"{folder_path}/ddpg.{time.time()}.pth"
        Toolbox.create_folder(model_path)
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
            model_path,
        )


class LearningTD3(LearningInterface, Generic[LearningTD3]):
    def __init__(
        self,
        env: gym.spaces.Box,
        actor: torch.nn.Module,
        target_actor: torch.nn.Module,
        critic1: torch.nn.Module,
        target_critic1: torch.nn.Module,
        critic2: torch.nn.Module,
        target_critic2: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        total_timesteps: int = 10_000,
        learning_starts: int = 32,
        batch_size: int = 32,
        policy_noise: float = 0.2,
        exploration_noise: float = 0.1,
        gamma: float = 0.99,
        policy_frequency: int = 2,
        noise_clip: float = 0.5,
        tau: float = 0.005,
        seed: Union[int, None] = None,
        device: torch.device = torch.device("cpu"),
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
        self.critic1 = critic1
        self.target_critic1 = target_critic1
        self.critic2 = critic2
        self.target_critic2 = target_critic2

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.replay_buffer = replay_buffer

        self.total_timesteps = total_timesteps
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.exploration_noise = exploration_noise
        self.gamma = gamma
        self.policy_frequency = policy_frequency
        self.noise_clip = noise_clip
        self.tau = tau
        self.seed = seed
        self.device = device

        self.critic_loss: List[float] = []
        self.actor_loss: List[float] = []

        self.predicted_observation: List[NDArray[np.float64]] = []
        self.predicted_action: List[NDArray[np.float64]] = []
        self.predicted_reward: List[NDArray[np.float64]] = []

    def train(self: LearningTD3, gradient_clipping: bool = False) -> LearningTD3:
        """
        Training Session for TD3.
        """
        if gradient_clipping:
            for p_critic1 in self.critic1.parameters():
                p_critic1.register_hook(self.soft_clipping)

            for p_critic2 in self.critic2.parameters():
                p_critic2.register_hook(self.soft_clipping)

            for p_actor in self.actor.parameters():
                p_actor.register_hook(self.soft_clipping)

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
                    actions += torch.normal(
                        0, torch.ones_like(actions) * self.exploration_noise
                    )
                    actions = (
                        actions.cpu()
                        .numpy()
                        .clip(
                            self.env.single_action_space.low,
                            self.env.single_action_space.high,
                        )
                    )

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
                    clipped_noise = (
                        torch.randn_like(data.actions, device=self.device)
                        * self.policy_noise
                    ).clamp(-self.noise_clip, self.noise_clip)
                    next_state_actions = (
                        self.target_actor(data.next_observations) + clipped_noise
                    )

                    critic1_next_target = self.target_critic1(
                        data.next_observations, next_state_actions
                    )
                    critic2_next_target = self.target_critic2(
                        data.next_observations, next_state_actions
                    )
                    min_critic_next_target = torch.min(
                        critic1_next_target, critic2_next_target
                    )
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.gamma * (min_critic_next_target).view(-1)

                critic1_a_values = self.critic1(data.observations, data.actions).view(
                    -1
                )
                critic2_a_values = self.critic2(data.observations, data.actions).view(
                    -1
                )
                critic1_loss = F.mse_loss(critic1_a_values, next_q_value)
                critic2_loss = F.mse_loss(critic2_a_values, next_q_value)
                critic_loss = critic1_loss + critic2_loss

                # optimize the model
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                if global_step % self.policy_frequency == 0:
                    actor_loss = -self.critic1(
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
                        self.critic1.parameters(), self.target_critic1.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.critic2.parameters(), self.target_critic2.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )

                if global_step % 100 == 0:
                    self.critic_loss.append(critic_loss.item())
                    self.actor_loss.append(actor_loss.item())

        self._last_called = self.train.__name__
        return self

    def save(self, folder_path: str) -> None:
        model_path = f"{folder_path}/td3.{time.time()}.pth"
        Toolbox.create_folder(model_path)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
            },
            model_path,
        )


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
                        wishart.rvs(
                            self.sample_dim, np.eye(self.sample_dim), size=2
                        ).reshape(-1, self.sample_dim << 2)
                        for _ in range(self.env.num_envs)
                    ]
                ).reshape(1, -1)
            else:
                with torch.no_grad():
                    actions = self.actor(torch.from_numpy(self.obs).to(self.device))
                    actions += torch.normal(
                        0, torch.ones_like(actions) * self.exploration_noise
                    )
                    actions = (
                        actions.cpu()
                        .numpy()
                        .clip(
                            self.env.single_action_space.low,
                            self.env.single_action_space.high,
                        )
                    )

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

                # for name, param in self.critic.named_parameters():
                #     if param.requires_grad:
                #         print(name, param.grad)

                self.critic_optimizer.step()

                if global_step % self.policy_frequency == 0:
                    actor_loss = -self.critic(
                        data.observations, self.actor(data.observations)
                    ).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    # torch.nn.utils.clip_grad_norm(self.actor.parameters(), float(1e-20))

                    # for name, param in self.actor.named_parameters():
                    #     if param.requires_grad:
                    #         print(name, param.grad)

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
                    self.actor_loss.append(actor_loss.item())

        self._last_called = self.train.__name__
        return self


class LearningTD3Random(LearningTD3):
    def train(self: LearningTD3, gradient_clipping: bool = False) -> LearningTD3:
        """
        Training Session for TD3.
        """
        if gradient_clipping:
            for p_critic1 in self.critic1.parameters():
                p_critic1.register_hook(self.soft_clipping)

            for p_critic2 in self.critic2.parameters():
                p_critic2.register_hook(self.soft_clipping)

            for p_actor in self.actor.parameters():
                p_actor.register_hook(self.soft_clipping)

        for global_step in trange(self.total_timesteps):
            if global_step < self.learning_starts:
                actions = np.array(
                    [
                        wishart.rvs(
                            self.sample_dim, np.eye(self.sample_dim), size=2
                        ).reshape(-1, self.sample_dim << 2)
                        for _ in range(self.env.num_envs)
                    ]
                ).reshape(1, -1)
            else:
                with torch.no_grad():
                    actions = self.actor(torch.from_numpy(self.obs).to(self.device))
                    actions += torch.normal(
                        0, torch.ones_like(actions) * self.exploration_noise
                    )
                    actions = (
                        actions.cpu()
                        .numpy()
                        .clip(
                            self.env.single_action_space.low,
                            self.env.single_action_space.high,
                        )
                    )

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
                    clipped_noise = (
                        torch.randn_like(data.actions, device=self.device)
                        * self.policy_noise
                    ).clamp(-self.noise_clip, self.noise_clip)
                    next_state_actions = (
                        self.target_actor(data.next_observations) + clipped_noise
                    )

                    critic1_next_target = self.target_critic1(
                        data.next_observations, next_state_actions
                    )
                    critic2_next_target = self.target_critic2(
                        data.next_observations, next_state_actions
                    )
                    min_critic_next_target = torch.min(
                        critic1_next_target, critic2_next_target
                    )
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.gamma * (min_critic_next_target).view(-1)

                critic1_a_values = self.critic1(data.observations, data.actions).view(
                    -1
                )
                critic2_a_values = self.critic2(data.observations, data.actions).view(
                    -1
                )
                critic1_loss = F.mse_loss(critic1_a_values, next_q_value)
                critic2_loss = F.mse_loss(critic2_a_values, next_q_value)
                critic_loss = critic1_loss + critic2_loss

                # optimize the model
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                if global_step % self.policy_frequency == 0:
                    actor_loss = -self.critic1(
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
                        self.critic1.parameters(), self.target_critic1.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.critic2.parameters(), self.target_critic2.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )

                if global_step % 100 == 0:
                    self.critic_loss.append(critic_loss.item())
                    self.actor_loss.append(actor_loss.item())

        self._last_called = self.train.__name__
        return self
