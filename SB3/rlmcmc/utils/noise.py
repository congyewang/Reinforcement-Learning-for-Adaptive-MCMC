from typing import Callable
import numpy as np
from numpy.typing import DTypeLike

import torch

from stable_baselines3.common.noise import ActionNoise


class RLMHNormalActionNoise(ActionNoise):
    """Outputs two different Gaussian noises.

    Args:
        mean (np.ndarray): Mean value of the noise
        sigma (np.ndarray): Scale of the noise (std here)
        dtype (DTypeLike): Type of the output noise
    """

    def __init__(
        self,
        action_dim: int,
        sample_noise_mean: np.ndarray,
        log_proposal_ratio_noise_mean: np.ndarray,
        sample_noise_sigma: np.ndarray,
        log_proposal_ratio_noise_sigma: np.ndarray,
        sample_noise_lengthscale: np.ndarray,
        log_proposal_ratio_noise_lengthscale: np.ndarray,
        dtype: DTypeLike = np.float32,
    ) -> None:
        assert action_dim > 1, ValueError("action_dim must be greater than 1.")
        self.action_dim = action_dim

        self._sample_noise_mu = sample_noise_mean
        self._log_proposal_ratio_noise_mu = log_proposal_ratio_noise_mean

        assert (sample_noise_sigma >= 0).all(), ValueError(
            "sample_noise_sigma must be greater than or equal to 0."
        )
        assert (log_proposal_ratio_noise_sigma >= 0).all(), ValueError(
            "log_proposal_ratio_noise_sigma must be greater than or equal to 0."
        )
        assert sample_noise_lengthscale >= 0, ValueError(
            "sample_noise_lengthscale must be greater than or equal to 0."
        )
        assert log_proposal_ratio_noise_lengthscale >= 0, ValueError(
            "log_proposal_ratio_noise_lengthscale must be greater than or equal to 0."
        )

        assert sample_noise_mean.shape[0] == action_dim - 1, ValueError(
            "sample_noise_mean must have shape (action_dim - 1,)."
        )
        assert log_proposal_ratio_noise_mean.shape[0] == 1, ValueError(
            "log_proposal_ratio_noise_mean must have shape (1,)."
        )
        assert sample_noise_sigma.shape[0] == action_dim - 1, ValueError(
            "sample_noise_sigma must have shape (action_dim - 1,)."
        )
        assert log_proposal_ratio_noise_sigma.shape[0] == 1, ValueError(
            "log_proposal_ratio_noise_sigma must have shape (1,)."
        )
        assert sample_noise_lengthscale.shape[0] == 1, ValueError(
            "sample_noise_lengthscale must have shape (1,)."
        )
        assert log_proposal_ratio_noise_lengthscale.shape[0] == 1, ValueError(
            "log_proposal_ratio_noise_lengthscale must have shape (1,)."
        )

        self._sample_noise_sigma = sample_noise_lengthscale * sample_noise_sigma
        self._log_proposal_ratio_noise_sigma = (
            log_proposal_ratio_noise_lengthscale * log_proposal_ratio_noise_sigma
        )

        self._dtype = dtype
        super().__init__()

    def __call__(self) -> np.ndarray:
        sample_noise = np.random.normal(
            self._sample_noise_mu, self._sample_noise_sigma
        ).astype(self._dtype)
        log_proposal_ratio_noise = np.random.normal(
            self._log_proposal_ratio_noise_mu, self._log_proposal_ratio_noise_sigma
        ).astype(self._dtype)
        return np.concatenate((sample_noise, log_proposal_ratio_noise))

    def __repr__(self) -> str:
        return f"RLMHNormalActionNoise(\n\tsample_noise_mu={self._sample_noise_mu},\n\tsample_noise_sigma={self._sample_noise_sigma},\n\tlog_proposal_ratio_noise_mu={self._log_proposal_ratio_noise_mu},\n\tlog_proposal_ratio_noise_sigma={self._log_proposal_ratio_noise_sigma}\n)"


class RLMHChi2CovNoise(ActionNoise):
    def __init__(self, noise_df: int = 2, dtype: DTypeLike = torch.Tensor) -> None:
        assert noise_df > 0 and isinstance(noise_df, int), ValueError(
            "noise_df must be an integre and greater than 0."
        )
        self.noise_df = noise_df

        self._dtype = dtype
        super().__init__()

    def __call__(self) -> Callable:
        sample_noise = torch.distributions.chi2.Chi2(self.noise_df)
        return sample_noise

    def __repr__(self) -> str:
        return f"RLMHChi2CovNoise(\n\tnoise_df={self.noise_df}\n)"


class RLMHHalfCauchyCovNoise(ActionNoise):
    def __init__(self, scale: float = 1.0, dtype: DTypeLike = torch.Tensor) -> None:
        assert scale > 0.0 and isinstance(scale, float), ValueError(
            "noise_df must be a float and greater than 0."
        )
        self.scale = scale

        self._dtype = dtype
        super().__init__()

    def __call__(self) -> Callable:
        sample_noise = torch.distributions.half_cauchy.HalfCauchy(self.scale)
        return sample_noise

    def __repr__(self) -> str:
        return f"RLMHHalfCauchyCovNoise(\n\tscale={self.scale}\n)"


class RLMHNormalLowRankVectorNoise(ActionNoise):
    def __init__(
        self,
        loc: torch.Tensor,
        covariance_matrix: torch.Tensor,
        dtype: DTypeLike = torch.Tensor,
    ) -> None:
        self.loc = loc
        self.covariance_matrix = covariance_matrix
        self._dtype = dtype
        super().__init__()

    def __call__(self) -> Callable:
        sample_noise = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=self.loc, covariance_matrix=self.covariance_matrix
        )
        return sample_noise

    def __repr__(self) -> str:
        return f"RLMHNormalLowRankVectorNoise(\n\tloc={self.loc}\n\tcovariance_matrix={self.covariance_matrix}\n)"
