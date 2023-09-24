import gymnasium as gym
from gymnasium import spaces

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad

INF = 3.4028235e+38 # Corresponds to the value of FLT_MAX in C++
SEED = 1234
generator = np.random.Generator(np.random.PCG64(SEED))


class MyEnv2D(gym.Env):
    def __init__(self, log_p, dim=2, max_steps=10_000):
        super(MyEnv2D, self).__init__()

        # Target Distribution
        self.log_p = log_p

        # Parameter
        self.dim = dim
        self.max_steps = max_steps
        self.ts = 0  # iteration time

        # Observation specification
        self.observation_space = spaces.Box(low=-INF, high=INF, shape=(dim,))

        # Action specification
        self.action_space = spaces.Box(low=0.0, high=INF, shape=(dim,))

        # Store
        self.store_state = []
        self.store_log_accetance_rate = []
        self.store_accetped_status = []
        self.store_action = []
        self.store_reward = []
        self.store_esjd = []
        self.store_esjd_error = []

    def env_mh(self, theta_curr, policy_func, log_p):
        cov_curr = policy_func(theta_curr) * np.eye(self.dim)

        theta_prop = generator.multivariate_normal(theta_curr, cov_curr)
        cov_prop = policy_func(theta_prop) * np.eye(self.dim)

        log_p_prop = log_p(theta_prop)
        log_p_curr = log_p(theta_curr)
        log_q_prop = multivariate_normal.logpdf(theta_prop, mean=theta_curr, cov=cov_curr)
        log_q_curr = multivariate_normal.logpdf(theta_curr, mean=theta_prop, cov=cov_prop)

        log_alpha = log_p_prop \
                - log_p_curr \
                + log_q_curr \
                - log_q_prop

        if np.log(generator.uniform()) < log_alpha:
            theta_curr = theta_prop
            accepted_status = True
        else:
            accepted_status = False

        return theta_curr, accepted_status, theta_prop, log_alpha

    def step(self, action, policy_func):
        sigma_curr = action

        # MCMC Environment
        state_curr, accepted_status, theta_prop, log_alpha = self.env_mh(theta_curr=self.state, policy_func=policy_func, log_p=self.log_p)

        # Store
        self.store_state.append(state_curr)
        self.store_accetped_status.append(accepted_status)
        self.store_action.append(sigma_curr)
        self.store_log_accetance_rate.append(log_alpha)

        # Calculate Reward
        reward = np.power(np.linalg.norm(self.state - state_curr, 2), 2)
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
            "theta_prop": theta_prop
        }

        return state_curr, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        self.ts = 0
        self.state = 10.0 * np.ones(self.dim)  # initialize s_{t}
        self.store_state.append(self.state)
        self.store_accetped_status.append(True)
        self.store_action.append(np.eye(self.dim))
        self.store_reward.append(0.0)
        self.store_log_accetance_rate.append(np.array([0.0]))

        # Information
        info = {
            "state": self.state,
            "accepted_status": True,
            "reward": 0.0
        }

        # Initialize ESJD Parameters
        # self.quartile()

        return self.state, info

    def render(self, mode="human"):
        plt.plot(self.store_state)
        plt.show()

    def log_q_proposal(self, x_t_plus_1, x_t, policy_func):
        sigma_t = policy_func(x_t)

        return multivariate_normal.logpdf(x_t_plus_1, loc=x_t, scale=sigma_t)

    def log_acceptance_ratio(self, x_t, x_t_plus_1, policy_func):
        sigma_t = policy_func(x_t)
        sigma_t_plus_1 = policy_func(x_t_plus_1)

        log_p_t_plus_1 = self.log_p(x_t_plus_1)
        log_p_t = self.log_p(x_t)
        log_q_t_plus_1 = multivariate_normal.logpdf(x_t_plus_1, loc=x_t, scale=sigma_t)
        log_q_t = multivariate_normal.logpdf(x_t, loc=x_t_plus_1, scale=sigma_t_plus_1)

        log_alpha = log_p_t_plus_1 \
                - log_p_t \
                + log_q_t \
                - log_q_t_plus_1

        return min(0, log_alpha)

    def log_squared_jump_distance(self, x_t, x_t_plus_1):
        return np.log(np.power(np.linalg.norm(np.array([x_t]) - np.array([x_t_plus_1]), 2), 2))

    def expected_squared_jump_distance_single_iteration(self, x_t, x_t_plus_1, policy_func):
        return np.exp(self.log_p(x_t) \
            + self.log_acceptance_ratio(x_t, x_t_plus_1, policy_func) \
            + self.log_q_proposal(x_t_plus_1, x_t, policy_func) \
            + self.log_squared_jump_distance(x_t, x_t_plus_1))

    # def quartile(self, loc=0, scale=1, alpha=0.005):
    #     self.lower_limit, self.upper_limit = multivariate_normal.ppf([0.5 * alpha, 1 - 0.5 * alpha], loc=loc, scale=scale)

    # def expected_squared_jump_distance(self, policy_func):
    #     """
    #     Numerical integration to calculate expected squared jump distance
    #     """
    #     result, error = dblquad(
    #         self.expected_squared_jump_distance_single_iteration,
    #         self.lower_limit,
    #         self.upper_limit,
    #         lambda x: self.lower_limit,
    #         lambda x: self.upper_limit,
    #         args=[policy_func]
    #         )

    #     self.store_esjd.append(result)
    #     self.store_esjd_error.append(error)

    #     # return result, error