import random
import time

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import plotly.graph_objects as go

from myenv import MyEnv
from distributions import Distribution

import toml
from types import SimpleNamespace

config = toml.load("config.toml")
args = SimpleNamespace(**config)


# Initialize Critic
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(24)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


# Initialize Actor
class Actor:
    def apply(self, params, observations):
        phi = jnp.arccos(params[0] + params[1]*observations[0] + params[2]*observations[1])
        alpha = params[3]**2 + params[4]**2 * (observations[0] - params[5])**2 + params[6]**2 * (observations[1] - params[7])**2
        beta = params[8]**2 + params[9]**2 * (observations[0] - params[10])**2 + params[11]**2 * (observations[1] - params[12])**2

        t1 = jnp.array([[jnp.cos(phi), -jnp.sin(phi)], [jnp.sin(phi), jnp.cos(phi)]])
        t2 = jnp.array([[alpha, 0], [0, beta]])

        sigma2 = t1 @ t2 @ t1.T

        return sigma2


class ActorTrainState(TrainState):
    params: jnp.ndarray
    target_params: jnp.ndarray

class TrainState(TrainState):
    params: flax.core.FrozenDict
    target_params: flax.core.FrozenDict


# Record the hyperparameters
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)


# Random seed
random.seed(args.seed)
np.random.seed(args.seed)
key = jax.random.PRNGKey(args.seed)
key, actor_key, qf1_key = jax.random.split(key, 3)


# Setup env
log_p = Distribution.gaussian
dim = 2
max_steps=100

env = MyEnv(log_p, dim, max_steps)
max_action = float(env.action_space.high[0])
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

actor = Actor()
qf1 = QNetwork()


actor_state = ActorTrainState.create(
    apply_fn=actor.apply,
    params=jnp.array([0.0, 0.0, 0.0, 1.0, 2.5, 0.0, 2.5, 0.0, 1.0, 2.5, 0.0, 2.5, 0.0]),
    target_params=jnp.array([0.0, 0.0, 0.0, 1.0, 2.5, 0.0, 2.5, 0.0, 1.0, 2.5, 0.0, 2.5, 0.0]),
    tx=optax.adam(
        learning_rate=args.learning_rate
        ),
)

qf1_state = TrainState.create(
    apply_fn=qf1.apply,
    params=qf1.init(qf1_key, obs, env.action_space.sample()),
    target_params=qf1.init(qf1_key, obs, env.action_space.sample()),
    tx=optax.adam(learning_rate=args.learning_rate),
)


actor.apply = jax.jit(jax.vmap(actor.apply, in_axes=(None, 0)))
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
    next_state_actions = (actor.apply(actor_state.target_params, next_observations)).clip(-1, 1)
    qf1_next_target = qf1.apply(qf1_state.target_params, next_observations, next_state_actions.reshape(args.batch_size, -1)).reshape(-1)
    next_q_value = (rewards + (1 - dones) * args.gamma * (qf1_next_target)).reshape(-1)

    def mse_loss(params):
        qf1_a_values = qf1.apply(params, observations, actions).squeeze()
        return ((qf1_a_values - next_q_value) ** 2).mean(), qf1_a_values.mean()

    (qf1_loss_value, qf1_a_values), grads = jax.value_and_grad(mse_loss, has_aux=True)(qf1_state.params)
    qf1_state = qf1_state.apply_gradients(grads=grads)
    return qf1_state, qf1_loss_value, qf1_a_values


# Inital the State of Optimizer
opt_state = actor_state.tx.init(actor_state.params)


def update_actor(actor_state, observations, qf1, qf1_state):

    def actor_loss(params, observations, qf1, qf1_state):
        actions = actor.apply(params, observations)
        return -qf1.apply(qf1_state.params, observations, actions.reshape(args.batch_size, -1)).mean()

    actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params, observations, qf1, qf1_state)
    actor_state = actor_state.apply_gradients(grads=grads)
    actor_state = actor_state.replace(
        target_params=optax.incremental_update(actor_state.params, actor_state.target_params, args.tau)
    )
    qf1_state = qf1_state.replace(
        target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, args.tau)
    )
    return actor_state, qf1_state, actor_loss_value


start_time = time.time()
for global_step in range(args.total_timesteps):
    # Action
    if global_step < args.learning_starts:
        actions = np.eye(env.dim).flatten()
        policy_cov_func = lambda obs: jnp.eye(env.dim)
    else:
        policy_cov_func = lambda obs: jnp.squeeze(actor.apply(actor_state.params, obs.reshape(1, -1)))
        actions = policy_cov_func(obs).flatten()

    # Execute the env and log data.
    actions_matrix = actions.reshape(env.dim, -1)
    next_obs, rewards, terminateds, truncateds, infos = env.step(actions_matrix, policy_cov_func)

    # Record rewards for plotting purposes
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            break

    # Save data to reply buffer; handle `terminal_observation`
    real_next_obs = next_obs.copy()
    rb.add(obs, real_next_obs, actions, rewards, terminateds, infos)

    # Update observation
    obs = next_obs

    # Training.
    if global_step > args.learning_starts:
        data = rb.sample(args.batch_size)
        qf1_state, qf1_loss_value, qf1_a_values = update_critic(
            actor_state,
            qf1_state,
            data.observations.numpy(),
            data.actions.numpy(),
            data.next_observations.numpy(),
            data.rewards.flatten().numpy(),
            data.dones.flatten().numpy(),
        )
        if global_step % args.policy_frequency == 0:
            actor_state, qf1_state, actor_loss_value = update_actor(
                actor_state,
                data.observations.numpy(),
                qf1,
                qf1_state,
            )

        if global_step % 100 == 0:
            writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
            writer.add_scalar("losses/actor_loss", actor_loss_value.item(), global_step)
            writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


# Plot Policy

x0 = np.linspace(-5., 5., 1000)
x1 = x0.copy()
x, y = np.meshgrid(x0, x1)

res = []
for i in range(1000):
    for j in range(1000):
        res.append(np.trace(policy_cov_func(np.array([x[i,j], y[i,j]]))))

z = np.array(res).reshape(1000, 1000)


fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
fig.update_layout(scene=dict(xaxis_title='x',
                             yaxis_title='y',
                             zaxis_title='Trace'))

fig.show()


