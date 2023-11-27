import gymnasium as gym
from gymnasium import spaces

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

INF = 3.4028235e+38 # Corresponds to the value of FLT_MAX in C++
SEED = 1234
generator = np.random.Generator(np.random.PCG64(SEED))


class MyEnv2DCov(gym.Env):
    def __init__(self, log_p, dim=2, max_steps=10_000):
        super(MyEnv2DCov, self).__init__()

        # Target Distribution
        self.log_p = log_p

        # Parameter
        self.dim = dim
        self.max_steps = max_steps
        self.ts = 0  # iteration time

        # Observation specification
        self.observation_space = spaces.Box(low=-INF, high=INF, shape=(1,dim))

        # Action specification
        self.action_space = spaces.Box(low=0.0, high=INF, shape=(1,int((1 + dim) * dim / 2 + 1)))

        # Store
        self.store_state = []
        self.store_log_accetance_rate = []
        self.store_accetped_status = []
        self.store_action = []
        self.store_reward = []

    def reshape_vector(self, v):
        v_flat = v.flatten()
        tri_matrix = np.zeros((self.dim, self.dim))
        index = 0
        for i in range(self.dim):
            for j in range(i + 1):
                tri_matrix[i, j] = v_flat[index]
                index += 1
        last_element = v_flat[-1]

        return tri_matrix, last_element

    def log_mvn(self, x, mean, cov):
        """Multivariate normal distribution."""
        return (-0.5 * len(mean) * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov)) - 0.5 * (x - mean) @ np.linalg.inv(cov) @ (x - mean).T).squeeze()

    def env_mh(self, theta_curr, policy_func, log_p):
        L_curr, mag_curr = self.reshape_vector(policy_func(theta_curr))
        cov_curr = L_curr @ L_curr.T + mag_curr * np.eye(self.dim)

        theta_prop = generator.multivariate_normal(theta_curr.flatten(), cov_curr)
        L_prop, mag_prop = self.reshape_vector(policy_func(theta_prop.reshape(1, -1)))
        cov_prop = L_prop @ L_prop.T + mag_prop * np.eye(self.dim)

        log_p_prop = log_p(theta_prop)
        log_p_curr = log_p(theta_curr)
        # log_q_prop = multivariate_normal.logpdf(theta_prop, mean=theta_curr.flatten(), cov=cov_curr, allow_singular=True)
        # log_q_curr = multivariate_normal.logpdf(theta_curr, mean=theta_prop.flatten(), cov=cov_prop, allow_singular=True)
        log_q_prop = self.log_mvn(theta_prop, theta_curr.flatten(), cov_curr)
        log_q_curr = self.log_mvn(theta_curr, theta_prop.flatten(), cov_prop)

        log_alpha = log_p_prop \
                - log_p_curr \
                + log_q_curr \
                - log_q_prop

        if np.log(generator.uniform()) < log_alpha:
            theta_curr = theta_prop
            accepted_status = True
        else:
            accepted_status = False

        return theta_curr.reshape(1, -1), accepted_status, theta_prop.reshape(1, -1), log_alpha

    def step(self, action, policy_func):
        sigma_curr = action

        # MCMC Environment
        state_curr, accepted_status, state_prop, log_alpha = self.env_mh(theta_curr=self.state, policy_func=policy_func, log_p=self.log_p)

        # Store
        self.store_state.append(state_curr)
        self.store_accetped_status.append(accepted_status)
        self.store_action.append(sigma_curr)
        self.store_log_accetance_rate.append(log_alpha)

        # Calculate Reward
        reward = np.power(np.linalg.norm(state_curr - state_prop, 2), 2) * np.exp(log_alpha)
        self.store_reward.append(reward)

        # Update Iteration Time
        self.state = state_curr
        self.ts += 1

        # Check for Completion
        terminated = self.ts >= self.max_steps
        truncated = terminated
        if terminated:
            # self.reset()
            pass

        # Information
        info = {
            "state": state_curr,
            "accepted_status": accepted_status,
            "reward": reward,
            "theta_prop": state_prop
        }

        return state_curr, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        self.ts = 0
        self.state = np.zeros(self.dim).reshape(1, -1)  # initialize s_{t}
        self.store_state.append(self.state)
        self.store_accetped_status.append(True)
        self.store_action.append(
            np.ones(
                int(
                    (1 + self.dim) * self.dim / 2 + 1
                    )
                ).reshape(1, -1)
            )
        self.store_reward.append(0.0)
        self.store_log_accetance_rate.append(np.array([0.0]))

        # Information
        info = {
            "state": self.state,
            "accepted_status": True,
            "reward": 0.0
        }

        return self.state, info

    def render(self, mode="human"):
        plt.plot(self.store_state)
        plt.show()


if __name__ == "__main__":
    log_p = lambda x: -x @ x.T / 2
    env = MyEnv2DCov(log_p=log_p, dim=2)
    obs, _ = env.reset()
    actions = np.array([1.0, 1.0, 1.0, 1.0]).reshape(1, -1)
    policy_func = lambda x: np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, -1)
    state_curr, reward, terminated, truncated, info = env.step(actions, policy_func)
    print(env.step(actions, policy_func))
