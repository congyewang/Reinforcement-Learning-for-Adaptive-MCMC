import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from gymnasium.vector import SyncVectorEnv


class QNetwork(nn.Module):
    def __init__(self, envs: SyncVectorEnv):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(envs.single_observation_space.shape).prod()
            + np.prod(envs.single_action_space.shape),
            32
        )
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc_out = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        x = torch.cat([x, a], 1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc_out(x)
        return x
