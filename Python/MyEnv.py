import torch
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from mcmc import rwm_env


class MyEnv(gym.Env):
    def __init__(self):
        super(MyEnv, self).__init__()
        # Parameter
        self.sigma = 1
        self.epsilon = 0.01
        self.nits = 1
        self.MaxSteps = 100
        self.Reward = 0
        self.Ts = 0  # iteration time
        self.State = 0  # state at this time, s_{t}
        self.OldState = 0  # state at previous state, s_{t-1}
        # Observation specification
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        # Action specification
        self.action_space = spaces.Box(low=0.0, high=99999.0, shape=(1,))
        self.log_pi = lambda x: -x**2/2

    def step(self, action):
        self.sigma = action[0]
        xt = rwm_env(sigma=self.sigma, theta_start=self.State, log_pi=self.log_pi, nits=self.nits)
        NextObs = self.State
        self.OldState = self.State  # Save s_{t-1}
        self.State = xt  # Update xt in this state

        # Calculate Reward
        Reward = torch.pow(torch.norm(self.State - self.OldState, 2),2)

        # Update Iteration Time
        self.Ts += 1

        # Check for Completion
        terminated = self.Ts >= self.MaxSteps
        truncated = terminated
        if terminated:
            self.reset()

        return NextObs, Reward, terminated, truncated, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.Ts = 0
        self.State = torch.Tensor([0.])  # initialize s_{t}
        self.OldState = torch.Tensor([0.])  # initialize s_{t-1}
        return self.State, {}
