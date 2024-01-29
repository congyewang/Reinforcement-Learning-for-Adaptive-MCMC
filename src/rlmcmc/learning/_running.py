from abc import ABC, abstractmethod

import os
import wandb
import dataclasses

import torch
import numpy as np
import matplotlib.pyplot as plt

from ..utils import MCMCAnimation, Toolbox
from . import LearningFactory

from typing import Union, Callable
from numpy.typing import NDArray


class RunningFactoryInterface(ABC):
    def __init__(
        self,
        project_name: str,
        posterior_name: str,
        posteriordb_path: str,
        compile: bool = False,
        mode: str = "ddpg",
    ) -> None:
        assert mode in [
            "ddpg",
            "td3",
            "ddpg_random",
        ], "mode must be ddpg, td3, or ddpg_random."

        self.project_name = project_name
        self.posterior_name = posterior_name
        self.posteriordb_path = posteriordb_path
        self.compile = compile

    def plot_action(
        self,
        log_target_pdf: Callable[[NDArray[np.float64]], np.float64],
        actor: torch.nn.Module,
        x_points: list[float],
        y_points: list[float],
        num: int = 1000,
        xlim: tuple[Union[float, int], Union[float, int]] = (-5.0, 5.0),
        ylim: tuple[Union[float, int], Union[float, int]] = (-5.0, 5.0),
        device: torch.device = torch.device("cpu"),
    ) -> None:
        fig, ax = plt.subplots(1, 1)

        # # Target distribution
        x_target = np.linspace(xlim[0], xlim[1], num)
        y_target = np.linspace(ylim[0], ylim[1], num)
        grid_x_target, grid_y_target = np.meshgrid(x_target, y_target)

        pdf_res = np.zeros((num, num))

        for i in range(len(x_target)):
            for j in range(len(y_target)):
                pdf_res[i, j] = np.exp(
                    log_target_pdf(np.array([x_target[i], y_target[j]]))
                )

        ax.contour(grid_x_target, grid_y_target, pdf_res.T)

        # Action distribution
        x = torch.tensor(x_points, dtype=torch.float64).to(device)
        y = torch.tensor(y_points, dtype=torch.float64).to(device)

        grid_x, grid_y = torch.meshgrid(x, y)

        samples = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
        mcmc_noise_zero = torch.zeros((samples.shape[0], 2))
        state = torch.cat((samples, mcmc_noise_zero), dim=1)
        with torch.no_grad():
            cov = actor(state).detach().numpy()[:, 0:4].reshape(-1, 2, 2)

        samples_np = samples.detach().numpy()

        for i in range(samples.shape[0]):
            Toolbox.plot_action(samples[i], cov[i], 1.0, ax)
            ax.plot(samples_np[i][0], samples_np[i][1], "kx")

        ax.axis([xlim[0] - 1.0, xlim[1] + 1.0, ylim[0] - 1.0, ylim[1] + 1.0])
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Action Plot")
        plt.tight_layout()
        wandb.log({"action_plot": wandb.Image(fig)})

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError("run method is not implemented.")


class RunningFactory(RunningFactoryInterface):
    def __init__(
        self,
        project_name: str,
        posterior_name: str,
        posteriordb_path: str,
        compile: bool = False,
        mode: str = "ddpg",
        **kwargs,
    ) -> None:
        super().__init__(project_name, posterior_name, posteriordb_path, compile)
        self.learning_factory = LearningFactory(
            posterior_name=self.posterior_name,
            posteriordb_path=self.posteriordb_path,
            compile=self.compile,
            mode=mode,
            **kwargs,
        )
        self.learning = self.learning_factory.create(mode=mode)

    def run(
        self,
        gradient_clipping: bool = True,
        x_points: list[float] = [-1.5, 0.0, 1.5],
        y_points: list[float] = [-1.5, 0.0, 1.5],
        xlim: tuple[Union[float, int], Union[float, int]] = (-5.0, 5.0),
        ylim: tuple[Union[float, int], Union[float, int]] = (-5.0, 5.0),
        save_path: str = "./save",
    ) -> None:
        wandb.init(
            project=self.project_name,
            config=dataclasses.asdict(self.learning_factory.args),
        )

        # Training
        training = self.learning.train(gradient_clipping=gradient_clipping)
        training.plot()
        wandb.log({"learning_data": wandb.Table(dataframe=training.dataframe())})

        training_animation = MCMCAnimation(
            log_target_pdf=self.learning_factory.log_target_pdf,
            dataframe=training.dataframe(),
            xlim=xlim,
            ylim=ylim,
        )
        training_animation_path = os.path.join(save_path, "animation/training.mp4")
        training_animation.make().save(training_animation_path, writer="ffmpeg")
        wandb.log(
            {
                "animation_traininig": wandb.Video(
                    training_animation_path, fps=1, format="mp4"
                )
            }
        )

        # Prediction
        prediction = self.learning.predict(
            predicted_env=self.learning_factory.predicted_envs
        )
        prediction.plot()
        wandb.log({"prediction_data": wandb.Table(dataframe=prediction.dataframe())})

        prediction_animation = MCMCAnimation(
            log_target_pdf=self.learning_factory.log_target_pdf,
            dataframe=prediction.dataframe(),
            xlim=xlim,
            ylim=ylim,
        )
        prediction_animation_path = os.path.join(save_path, "animation/prediction.mp4")
        prediction_animation.make().save(prediction_animation_path, writer="ffmpeg")
        wandb.log(
            {
                "animation_prediction": wandb.Video(
                    prediction_animation_path, fps=1, format="mp4"
                )
            }
        )

        # Plot Action
        if self.learning_factory.args.sample_dim == 2:
            self.plot_action(
                log_target_pdf=self.learning_factory.log_target_pdf,
                actor=self.learning.actor,
                x_points=x_points,
                y_points=y_points,
                xlim=xlim,
                ylim=ylim,
                device=self.learning_factory.device,
            )

        # Save Model
        self.learning.save(os.path.join(save_path, "model"))

        wandb.finish()
