from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import numpy as np

import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer

import time
from tqdm.auto import trange
from typing import Union, List
from numpy.typing import NDArray


class LearningBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError("train method is not implemented")

    @abstractmethod
    def predict(self, predicted_timesteps: int) -> None:
        raise NotImplementedError("predict method is not implemented")

    @abstractmethod
    def save(self, folder_path: str) -> None:
        raise NotImplementedError("save method is not implemented")


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
        self.env = env
        self.obs, self.infos = env.reset(seed=seed)
        self.sample_dim: int = np.prod(env.single_observation_space.shape) >> 1
        assert isinstance(
            self.env.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"

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

    def predict(self, predicted_timesteps: int = 10_000) -> None:
        # Store predicted obs, action, and reward
        predicted_observation: List[NDArray[np.float64]] = []
        predicted_action: List[NDArray[np.float64]] = []
        predicted_reward: List[NDArray[np.float64]] = []

        # Reset the environment
        predicted_obs, _ = self.env.reset(seed=self.seed)

        for _ in trange(predicted_timesteps):
            with torch.no_grad():
                predicted_actions = self.actor(
                    torch.from_numpy(predicted_obs).to(self.device)
                )

            predicted_obs, predicted_rewards, _, _, _ = self.env.step(predicted_actions)

            predicted_observation.append(predicted_obs)
            predicted_action.append(predicted_actions.view(-1).detach().cpu().numpy())
            predicted_reward.append(predicted_rewards)

        self.predicted_observation = np.array(predicted_observation).reshape(
            -1, np.prod(self.env.single_observation_space.shape)
        )
        self.predicted_action = np.array(predicted_action)
        self.predicted_reward = np.array(predicted_reward).flatten()

    def save(self, folder_path: str) -> None:
        model_path = f"{folder_path}/ddpg.{time.time()}.pth"
        torch.save((self.actor.state_dict(), self.critic.state_dict()), model_path)


class LearningTD3(LearningBase):
    pass
