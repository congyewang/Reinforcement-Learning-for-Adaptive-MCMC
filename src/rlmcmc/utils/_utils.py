import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, MovieWriter
import gymnasium as gym
from dataclasses import dataclass
from scipy.stats._multivariate import _PSD
from scipy.stats import multivariate_normal as mvn

import json
import bridgestan as bs
from posteriordb import PosteriorDatabase

from typing import Callable, Dict, List, Tuple, Union
from numpy.typing import NDArray
from gymnasium.envs.registration import EnvSpec


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
    def make_env(
        env_id: Union[str, EnvSpec],
        seed: int,
        log_target_pdf: Callable,
        sample_dim: int,
        total_timesteps: int,
    ):
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
    def nearestPD(cls, A: NDArray[np.float64]):
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
    def isPD(cls, B: NDArray[np.float64]):
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

    @staticmethod
    def create_folder(file_path: str) -> None:
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


class MCMCAnimation:
    def __init__(
        self,
        log_target_pdf: Callable[NDArray[np.float64], np.float64],
        dataframe: pd.DataFrame,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        ellipse_num: int = 10,
        target: bool = True,
    ) -> None:
        self.log_target_pdf = log_target_pdf
        self.dataframe = dataframe.reset_index(drop=True)
        self.xlim = xlim
        self.ylim = ylim

        self.target = target
        self.ellipse_num = ellipse_num

        self.fig, self.ax = plt.subplots()
        self.ellipse_list = []  # List to track the ellipses

        self.setup_plot()

    def create_cov_ellipse(
        self,
        covariance: NDArray[np.float64],
        position: NDArray[np.float64],
        nstd: float = 2.0,
        **kwargs,
    ):
        """
        Create a covariance ellipse based on the covariance matrix and position.
        """
        eig_vals, eig_vecs = np.linalg.eigh(covariance)
        angle = np.degrees(np.arctan2(*eig_vecs[:, 0][::-1]))
        width, height = 2 * nstd * np.sqrt(eig_vals)
        ellipse = patches.Ellipse(position, width, height, angle, **kwargs)
        return ellipse

    def plot_target_distribution(self, num: int = 1_000):
        """
        Plot target distribution.
        """
        x = np.linspace(self.xlim[0], self.xlim[1], num)
        y = np.linspace(self.ylim[0], self.ylim[1], num)
        grid_x, grid_y = np.meshgrid(x, y)

        pdf_res = np.zeros((num, num))

        for i in range(len(x)):
            for j in range(len(y)):
                pdf_res[i, j] = np.exp(self.log_target_pdf(np.array([x[i], y[j]])))

        self.ax.contour(grid_x, grid_y, pdf_res.T)

    def setup_plot(self):
        """
        Setup the plot for the animation.
        """
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        if self.target:
            self.plot_target_distribution()

        # Initialize elements in the plot for the animation
        (self.accepted_trace,) = self.ax.plot(
            [], [], "o-", markersize=3, color="black", alpha=0.3, label="Accepted Trace"
        )  # Line trace for accepted samples
        (self.current_point,) = self.ax.plot(
            [], [], "o", markerfacecolor="black", markersize=10, label="Current Point"
        )  # Solid black point for the current accepted position
        (self.proposed_point,) = self.ax.plot(
            [], [], "o", markerfacecolor="none", markersize=10, markeredgecolor="red"
        )  # Solid red point for the proposed position

    def update(self, frame: int):
        """
        Function to update the animation for each frame
        """
        # Remove the oldest ellipse if more than ellipse_num ellipses are present
        if len(self.ellipse_list) > self.ellipse_num:
            self.ellipse_list.pop(0).remove()

        # Getting data for the current frame
        row = self.dataframe.iloc[frame]
        cov_matrix = [[row["cov1"], row["cov2"]], [row["cov3"], row["cov4"]]]

        # Update the trace of accepted samples
        self.accepted_trace.set_data(
            self.dataframe["x"][: frame + 1], self.dataframe["y"][: frame + 1]
        )

        # Update the current accepted point and the proposed point
        self.current_point.set_data(
            self.dataframe["x"][frame], self.dataframe["y"][frame]
        )
        self.proposed_point.set_data(row["proposed_x"], row["proposed_y"])

        # Draw the covariance ellipse for the current position
        ellipse = self.create_cov_ellipse(
            cov_matrix,
            (row["x"], row["y"]),
            edgecolor="blue",
            facecolor="none",
            alpha=0.1,
        )
        self.ax.add_patch(ellipse)
        self.ellipse_list.append(ellipse)

        # Set the title of the plot
        self.ax.set_title(f"2D MCMC Trajectory Animation - Iteration: {frame + 1}")

        return [self.accepted_trace, self.current_point, self.proposed_point, ellipse]

    def make(self, interval: int = 100, blit: bool = True, repeat: bool = False):
        """
        Creating the animation with the trace of accepted samples
        """
        self.anim = FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.dataframe),
            interval=interval,
            blit=blit,
            repeat=repeat,
        )
        return self

    def save(
        self, anim_file_path: str, writer: Union[MovieWriter, str, None] = "ffmpeg"
    ):
        """
        Save the animation
        """
        Toolbox.create_folder(anim_file_path)
        self.anim.save(anim_file_path, writer=writer)
