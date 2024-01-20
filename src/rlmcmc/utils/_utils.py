import re
import numpy as np
import matplotlib
from matplotlib.patches import Ellipse
import gymnasium as gym
from dataclasses import dataclass
from scipy.stats._multivariate import _PSD
from scipy.stats import multivariate_normal as mvn

import json
import bridgestan as bs
from posteriordb import PosteriorDatabase

from typing import Callable, Dict, List, Union
from numpy.typing import NDArray


@dataclass
class Args:
    exp_name: str = "test"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "RLMHEnv-v3.1"
    """the environment id of the Atari game"""
    total_timesteps: int = int(1e3)
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 24
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 4
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    sample_dim: int = 2
    log_target_pdf: Callable[
        [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], np.float64
    ] = lambda x: mvn.logpdf(x, mean=np.zeros(2), cov=np.eye(2))


class Toolbox:
    @staticmethod
    def make_env(env_id, seed, log_target_pdf, sample_dim, total_timesteps):
        def thunk():
            env = gym.make(
                env_id,
                log_target_pdf=log_target_pdf,
                sample_dim=sample_dim,
                total_timesteps=total_timesteps,
            )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)

            return env

        return thunk

    @classmethod
    def nearestPD(cls, A):
        """
        Find the nearest positive-definite matrix to input
        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """

        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if cls.isPD(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1
        while not cls.isPD(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3

    @classmethod
    def isPD(cls, B):
        """
        Returns true when input is positive-definite, via Cholesky, det, and _PSD from scipy.
        """
        try:
            _ = np.linalg.cholesky(B)
            res_cholesky = True
        except np.linalg.LinAlgError:
            res_cholesky = False

        try:
            assert np.linalg.det(B) > 0, "Determinant is negative"
            res_det = True
        except AssertionError:
            res_det = False

        try:
            _PSD(B, allow_singular=False)
            res_PSD = True
        except Exception as e:
            if re.search("[Pp]ositive", str(e)):
                return False
            else:
                raise

        res = res_cholesky and res_det and res_PSD

        return res

    @staticmethod
    def make_log_target_pdf(
        posterior_name: str,
        posteriordb_path: str,
        data: Union[Dict[str, Union[float, int]], None] = None,
    ):
        # Load DataBase Locally
        pdb = PosteriorDatabase(posteriordb_path)

        # Load Dataset
        posterior = pdb.posterior(posterior_name)
        stan_code = posterior.model.stan_code_file_path()
        if data is None:
            stan_data = json.dumps(posterior.data.values())
        else:
            stan_data = json.dumps(data)

        # Return log_target_pdf
        model = bs.StanModel.from_stan_file(stan_code, stan_data)

        return model.log_density

    @staticmethod
    def plot_action(
        x: Union[List[Union[float, int]], NDArray[np.float64]],
        a: NDArray[np.float64],
        msd: Union[float, np.float64],
        ax: matplotlib.axes.Axes,
    ) -> None:
        l, v = np.linalg.eig(a)
        wh = msd * np.sqrt(l)
        t = np.arctan2(v[1, 0], v[0, 0])
        deg = t * (180 / np.pi)
        ell = Ellipse(
            (x[0], x[1]),
            width=wh[0],
            height=wh[1],
            angle=deg,
            edgecolor="r",
            facecolor="none",
            alpha=0.7,
        )
        ax.add_patch(ell)
