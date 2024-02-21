from gymnasium.envs.registration import register
from ._env import (
    RLMHEnvBase,
    RLMHEnvV300a1,
    RLMHEnvV301a1,
    RLMHEnvV31,
    RLMHEnvV31A,
    RLMHEnvV31B,
    RLMHEnvV33,
    RLMHEnvV6,
    RLMHEnvV61,
    RLMHEnvV7,
    RLMHEnvV71,
    RLMHEnvV8,
    RLMHEnvV81,
    RLMHEnvV82,
    RLMHEnvV83,
    RLMHEnvV84,
)

register(
    id="RLMHEnv-v3.0.0.alpha.1",
    entry_point="src.rlmcmc.env._env:RLMHEnvV300a1",
)

register(
    id="RLMHEnv-v3.0.1.alpha.1",
    entry_point="src.rlmcmc.env._env:RLMHEnvV301a1",
)

register(
    id="RLMHEnv-v3.1",
    entry_point="src.rlmcmc.env._env:RLMHEnvV31",
)

register(
    id="RLMHEnv-v3.1.a",
    entry_point="src.rlmcmc.env._env:RLMHEnvV31A",
)

register(
    id="RLMHEnv-v3.1.b",
    entry_point="src.rlmcmc.env._env:RLMHEnvV31B",
)

register(
    id="RLMHEnv-v3.3",
    entry_point="src.rlmcmc.env._env:RLMHEnvV33",
)

register(
    id="RLMHEnv-v6.0",
    entry_point="src.rlmcmc.env._env:RLMHEnvV6",
)

register(
    id="RLMHEnv-v6.1",
    entry_point="src.rlmcmc.env._env:RLMHEnvV61",
)

register(
    id="RLMHEnv-v7.0",
    entry_point="src.rlmcmc.env._env:RLMHEnvV7",
)

register(
    id="RLMHEnv-v7.1",
    entry_point="src.rlmcmc.env._env:RLMHEnvV71",
)

register(
    id="RLMHEnv-v8.0",
    entry_point="src.rlmcmc.env._env:RLMHEnvV8",
)

register(
    id="RLMHEnv-v8.1",
    entry_point="src.rlmcmc.env._env:RLMHEnvV81",
)

register(
    id="RLMHEnv-v8.2",
    entry_point="src.rlmcmc.env._env:RLMHEnvV82",
)

register(
    id="RLMHEnv-v8.3",
    entry_point="src.rlmcmc.env._env:RLMHEnvV83",
)

register(
    id="RLMHEnv-v8.4",
    entry_point="src.rlmcmc.env._env:RLMHEnvV84",
)


__all__ = [
    "RLMHEnvBase",
    "RLMHEnvV300a1",
    "RLMHEnvV301a1",
    "RLMHEnvV31",
    "RLMHEnvV31A",
    "RLMHEnvV31B",
    "RLMHEnvV33",
    "RLMHEnvV6",
    "RLMHEnvV61",
    "RLMHEnvV7",
    "RLMHEnvV71",
    "RLMHEnvV8",
    "RLMHEnvV81",
    "RLMHEnvV82",
    "RLMHEnvV83",
    "RLMHEnvV84",
]
