import numpy as np

import gymnasium as gym

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback


gym.register(id='MyEnv-v0', entry_point='MyEnv:MyEnv')
env = gym.make('MyEnv-v0')
env.reset()

# Creating the model
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

# Create the callback: check every 1000 steps
callback = CheckpointCallback(save_freq=1000, save_path='./')

# Training the model
model.learn(total_timesteps=1000, callback=callback)

# Save the model
model.save("ddpg_myenv")
