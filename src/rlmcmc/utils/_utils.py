import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from scipy.stats import multivariate_normal as mvn

from typing import Callable
from numpy.typing import NDArray


@dataclass
class Args:
    exp_name: str = "test"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "RLMHEnv-v3.1"
    """the environment id of the Atari game"""
    total_timesteps: int = int(1e3)
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e3)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 24
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 4
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    sample_dim: int = 2
    log_target_pdf: Callable[
        [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], np.float64
    ] = lambda x: mvn.logpdf(x, mean=np.zeros(2), cov=np.eye(2))


class Toolbox:
    @staticmethod
    def make_env(env_id, seed, log_target_pdf, sample_dim, total_timesteps):
        def thunk():
            env = gym.make(
                env_id,
                log_target_pdf=log_target_pdf,
                sample_dim=sample_dim,
                total_timesteps=total_timesteps,
            )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)

            return env

        return thunk
