from gymnasium.envs.registration import register
from ._env import RLMHEnvBase, RLMHEnvV31, RLMHEnvV31A, RLMHEnvV31B, RLMHEnvV33


register(
    id='RLMHEnv-v3.1',
    entry_point='src.rlmcmc.env._env:RLMHEnvV31',
)

register(
    id='RLMHEnv-v3.1.a',
    entry_point='src.rlmcmc.env._env:RLMHEnvV31A',
)

register(
    id='RLMHEnv-v3.1.b',
    entry_point='src.rlmcmc.env._env:RLMHEnvV31B',
)

register(
    id='RLMHEnv-v3.3',
    entry_point='src.rlmcmc.env._env:RLMHEnvV33',
)

__all__ = ["RLMHEnvBase", "RLMHEnvV31", "RLMHEnvV31A", "RLMHEnvV31B", "RLMHEnvV33"]