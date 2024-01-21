import numpy as np
from functools import partial
from numpy.typing import NDArray

from ..rlmcmc.env import RLMHEnvV31, RLMHEnvV31A, RLMHEnvV31B


def multivariate_normal_logpdf(
    x: NDArray[np.float64],
    mean: NDArray[np.float64],
    cov: NDArray[np.float64],
) -> NDArray[np.float64]:
    return (
        -0.5 * len(mean) * np.log(2 * np.pi)
        - 0.5 * np.log(np.linalg.det(cov))
        - 0.5 * (x - mean) @ np.linalg.inv(cov) @ (x - mean).T
    )


class TestRLMHEnvV31:
    env_v31 = RLMHEnvV31(
        log_target_pdf=partial(
            multivariate_normal_logpdf,
            mean=np.array([10.0, 0.2]),
            cov=np.array([[4.0, 0.0], [0.0, 5.0]]),
        ),
        sample_dim=2,
        total_timesteps=1000,
    )

    def test_log_target_pdf(self):
        x = np.array([1.0, 2.1])
        mean = np.array([10.0, 0.2])
        cov = np.array([[4.0, 0.0], [0.0, 5.0]])

        assert self.env_v31.log_target_pdf(x) == multivariate_normal_logpdf(
            x, mean, cov
        ), "log_target_pdf not working"

    def test_distance_function(self):
        current_sample = np.array([12.0, -231.0])
        proposed_sample = np.array([-0.97, 82.0])

        assert self.env_v31.distance_function(
            current_sample, proposed_sample
        ) == np.sqrt(
            (12.0 - (-0.97)) ** 2 + (-231.0 - 82.0) ** 2
        ), "distance_function not working"

    def reward_function(self):
        current_sample = np.array([-0.48, 35.0])
        proposed_sample = np.array([71.0, -0.62])
        log_alpha = np.float64(-0.309)

        assert self.env_v31.reward_function(
            current_sample, proposed_sample, log_alpha
        ) == (np.sqrt((-0.48 - 71.0) ** 2 + (35.0 + 0.62) ** 2)) ** 2 * np.exp(
            -0.309
        ), "reward_function not working"

    def test_step(self):
        pass
