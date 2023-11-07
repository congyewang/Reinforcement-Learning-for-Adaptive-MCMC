from gymnasium.envs.registration import register
from .rl_mh import RLMHEnv

register(
    id='RLMHEnv-v0',
    entry_point='rl_mcmc_env.rl_mh:RLMHEnv',
)
