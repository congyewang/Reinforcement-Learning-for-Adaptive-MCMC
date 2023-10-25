import random

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer

from tqdm.auto import trange

from rl_env.myenv_2d_cov import MyEnv2DCov
from base_rl_mcmc.toolbox import flat

import toml
from types import SimpleNamespace

import os
import json
import pandas as pd
import bridgestan as bs
from posteriordb import PosteriorDatabase


config = toml.load("./base_rl_mcmc/config/config_ddpg.toml")
args = SimpleNamespace(**config)

# seeding
random.seed(args.seed)
np.random.seed(args.seed)
key = jax.random.PRNGKey(args.seed)
key, actor_key, qf1_key = jax.random.split(key, 3)


# Initialize Agent

class QNetwork(nn.Module):
    """
    Critic Network
    """
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(48)(x)
        x = nn.softplus(x)
        x = nn.Dense(48)(x)
        x = nn.softplus(x)
        x = nn.Dense(1)(x)
        x = nn.softplus(x)
        return x

class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        x = self.phi(obs, name="x_sigma")
        x_sigma = x[:,:-1]
        mag_sigma = x[:,-1].reshape(-1,1)
        mag_sigma = nn.softplus(mag_sigma)
        return jnp.concatenate([x_sigma, mag_sigma], -1)

    def phi(self, input, name):
        x = nn.Dense(48, name=f"{name}_dense1")(input)
        x = nn.softplus(x)
        x = nn.Dense(48, name=f"{name}_dense4")(x)
        x = nn.softplus(x)
        x = nn.Dense(int((1 + self.action_dim) * self.action_dim / 2 + 1), name=f"{name}_dense5")(x)
        x = nn.softplus(x)
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


# Load DataBase Locally
pdb_path = os.path.join(os.getcwd(), "posteriordb/posterior_database")
my_pdb = PosteriorDatabase(pdb_path)

# Load Dataset
posterior = my_pdb.posterior("test-SimpleEggBox-test-SimpleEggBox")
stan = posterior.model.stan_code_file_path()
data = json.dumps(posterior.data.values())
model = bs.StanModel.from_stan_file(stan, data)

# Gold Standard
gs = posterior.reference_draws()
df = pd.DataFrame(gs)
gs_chains = np.zeros((sum(flat(posterior.information['dimensions'].values())),\
                       posterior.reference_draws_info()['diagnostics']['ndraws']))
for i in range(len(df.keys())):
    s = []
    for j in range(len(df[df.keys()[i]])):
        s += df[df.keys()[i]][j]
    gs_chains[i, :] = s
linv = np.linalg.inv(np.cov(gs_chains))

# Extract log-P-pdf and its gradient
log_p = model.log_density
grad_log_p = lambda x: model.log_density_gradient(x)[1]
hess_log_p = lambda x: model.log_density_hessian(x)[2]



# Setup env
dim = 2
max_steps=100_000
args.total_timesteps = max_steps
args.batch_size = 48

env = MyEnv2DCov(log_p, dim, max_steps)
max_action = float(env.action_space.high[0,0])
env.observation_space.dtype = np.float32
rb = ReplayBuffer(
    args.buffer_size,
    env.observation_space,
    env.action_space,
    device='cpu',
    handle_timeout_termination=False,
)


# Start
obs, _ = env.reset()
obs = obs.reshape(-1, dim)
actor = Actor(action_dim=2)
qf1 = QNetwork()

actor_state = TrainState.create(
    apply_fn=actor.apply,
    params=actor.init(actor_key, obs),
    target_params=actor.init(actor_key, obs),
    tx=optax.adam(learning_rate=args.learning_rate),
)

qf1_state = TrainState.create(
    apply_fn=qf1.apply,
    params=qf1.init(qf1_key, obs, env.action_space.sample()),
    target_params=qf1.init(qf1_key, obs, env.action_space.sample()),
    tx=optax.adam(learning_rate=args.learning_rate),
)

actor.apply = jax.jit(actor.apply)
qf1.apply = jax.jit(qf1.apply)


@jax.jit
def update_critic(
    actor_state: TrainState,
    qf1_state: TrainState,
    observations: np.ndarray,
    actions: np.ndarray,
    next_observations: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
):
    next_state_actions = (actor.apply(actor_state.target_params, next_observations)).clip(-1, 1)  # TODO: proper clip
    qf1_next_target = qf1.apply(qf1_state.target_params, next_observations, next_state_actions).reshape(-1)
    next_q_value = (rewards + (1 - dones) * args.gamma * (qf1_next_target)).reshape(-1)

    def mse_loss(params):
        qf1_a_values = qf1.apply(params, observations, actions).squeeze()
        return ((qf1_a_values - next_q_value) ** 2).mean(), qf1_a_values.mean()

    (qf1_loss_value, qf1_a_values), grads = jax.value_and_grad(mse_loss, has_aux=True)(qf1_state.params)
    qf1_state = qf1_state.apply_gradients(grads=grads)
    return qf1_state, qf1_loss_value, qf1_a_values

@jax.jit
def update_actor(
    actor_state: TrainState,
    qf1_state: TrainState,
    observations: np.ndarray,
):
    def actor_loss(params):
        return -qf1.apply(qf1_state.params, observations, actor.apply(params, observations)).mean()

    actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=grads)
    actor_state = actor_state.replace(
        target_params=optax.incremental_update(actor_state.params, actor_state.target_params, args.tau)
    )
    qf1_state = qf1_state.replace(
        target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, args.tau)
    )
    return actor_state, qf1_state, actor_loss_value



for global_step in trange(args.total_timesteps):

    actions = actor.apply(actor_state.params, obs)
    actions = np.array(
        [
            (actions + np.random.normal(0, args.exploration_noise)).clip(
                env.action_space.low, env.action_space.high
            )
        ]
    )

    next_obs, rewards, terminateds, truncateds, infos = env.step(actions, lambda x: actor.apply(actor_state.params, x))

    real_next_obs = next_obs.copy()
    rb.add(obs, real_next_obs, actions, rewards, terminateds, infos)

    obs = next_obs

    # Training
    if global_step > args.learning_starts:
        data = rb.sample(args.batch_size)
        qf1_state, qf1_loss_value, qf1_a_values = update_critic(
            actor_state=actor_state,
            qf1_state=qf1_state,
            observations=data.observations.reshape(-1, dim).numpy(),
            actions=data.actions.numpy(),
            next_observations=data.next_observations.reshape(-1, dim).numpy(),
            rewards=data.rewards.flatten().numpy(),
            dones=data.dones.flatten().numpy()
        )
        if global_step % args.policy_frequency == 0:
            actor_state, qf1_state, actor_loss_value = update_actor(
                actor_state=actor_state,
                qf1_state=qf1_state,
                observations=data.observations.reshape(-1, dim).numpy(),
            )


state_list = np.array([i for i in env.store_state]).reshape(-1, dim)
action_list = np.array([i.squeeze() for i in env.store_action])

np.savetxt('SimpleEggBox_DDPG_small.csv', state_list, delimiter=',')
