import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            24,
        )
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
