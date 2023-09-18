import gymnasium as gym
from gymnasium import spaces

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import dblquad

INF = 3.4028235e+38 # Corresponds to the value of FLT_MAX in C++
SEED = 1234
generator = np.random.Generator(np.random.PCG64(SEED))


class MyEnv(gym.Env):
    def __init__(self, log_p, dim=1, max_steps=10_000):
        super(MyEnv, self).__init__()

        # Target Distribution
        self.log_p = log_p

        # Parameter
        self.dim = dim
        self.max_steps = max_steps
        self.ts = 0  # iteration time

        # Observation specification
        self.observation_space = spaces.Box(low=-INF, high=INF, shape=(dim,))

        # Action specification
        self.action_space = spaces.Box(low=-INF, high=INF, shape=(dim**2,))

        # Store
        self.store_state = []
        self.store_log_accetance_rate = []
        self.store_accetped_status = []
        self.store_action = []
        self.store_reward = []

    def env_mh(self, theta_curr, policy_func, log_p):
        sigma_curr = policy_func(theta_curr)

        theta_prop = generator.normal(theta_curr, sigma_curr).flatten()
        sigma_prop = policy_func(theta_prop)

        log_p_prop = log_p(theta_prop)
        log_p_curr = log_p(theta_curr)
        log_q_prop = norm.logpdf(theta_prop, loc=theta_curr, scale=sigma_curr)
        log_q_curr = norm.logpdf(theta_curr, loc=theta_prop, scale=sigma_prop)

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
        return self.state, info

    def render(self, mode="human"):
        plt.plot(self.store_state)
        plt.show()

    def log_q_proposal(self, x_t_plus_1, x_t, policy_func):
        sigma_curr = policy_func(x_t)

        return norm.logpdf(x_t_plus_1, loc=x_t, scale=sigma_curr)

    def log_acceptance_ratio(self, x_t, x_t_plus_1, policy_func):
        sigma_t = policy_func(x_t)
        sigma_t_plus_1 = policy_func(x_t_plus_1)

        log_p_t_plus_1 = self.log_p(x_t_plus_1)
        log_p_t = self.log_p(x_t)
        log_q_t_plus_1 = norm.logpdf(x_t_plus_1, loc=x_t, scale=sigma_t)
        log_q_t = norm.logpdf(x_t, loc=x_t_plus_1, scale=sigma_t_plus_1)

        log_alpha = log_p_t_plus_1 \
                - log_p_t \
                + log_q_t \
                - log_q_t_plus_1

        return min(0, log_alpha)

    def log_squared_jump_distance(self, x_t, x_t_plus_1):
        return np.log(np.power(np.linalg.norm(x_t - x_t_plus_1, 2), 2))

    def expected_squared_jump_distance_single_iteration(self, x_t, x_t_plus_1, policy_func):
        return np.exp(self.log_p(x_t) \
            + self.log_acceptance_ratio(x_t, x_t_plus_1, policy_func) \
            + self.log_q_proposal(x_t_plus_1, x_t) \
            + self.log_squared_jump_distance(x_t, x_t_plus_1))

    def numerical_integration(
            self,
            x_t_lower_bound=-np.inf,
            x_t_upper_bound=np.inf,
            x_t_plus_1_lower_bound=-np.inf,
            x_t_plus_1_upper_bound=np.inf
            ):
        """
        Numerical integration to calculate expected squared jump distance
        """
        result, _ = dblquad(
            self.expected_squared_jump_distance_single_iteration,
            x_t_lower_bound, x_t_upper_bound,  # x_t limits
            x_t_plus_1_lower_bound, x_t_plus_1_upper_bound,  # x_t_plus_1 limits
            lambda x_t_plus_1: x_t_lower_bound, lambda x_t_plus_1: x_t_upper_bound)  # x limits

        return result
