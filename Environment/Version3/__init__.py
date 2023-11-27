from gymnasium.envs.registration import register
from .rl_mh import RLMHEnv

register(
    id='RLMHEnv-v3',
    entry_point='Environment.Version3.rl_mh:RLMHEnv',
)

__all__ = ["RLMHEnv"]
