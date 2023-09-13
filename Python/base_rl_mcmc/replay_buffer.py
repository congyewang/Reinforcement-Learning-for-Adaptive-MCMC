import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer = {
            'state': np.zeros((capacity, state_dim), dtype=np.float32),
            'action': np.zeros((capacity, action_dim), dtype=np.float32),
            'reward': np.zeros(capacity, dtype=np.float32),
            'next_state': np.zeros((capacity, state_dim), dtype=np.float32),
            'done': np.zeros(capacity, dtype=np.bool_),
        }
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.buffer['state'][self.position] = state
        self.buffer['action'][self.position] = action
        self.buffer['reward'][self.position] = reward
        self.buffer['next_state'][self.position] = next_state
        self.buffer['done'][self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            'state': self.buffer['state'][indices],
            'action': self.buffer['action'][indices],
            'reward': self.buffer['reward'][indices],
            'next_state': self.buffer['next_state'][indices],
            'done': self.buffer['done'][indices],
        }

    def __len__(self):
        return self.size
