import importlib
import sys

import bridgestan as bs
import h5py
import matplotlib.pyplot as plt
import numpy as np
import stan
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import loadmat, savemat

np.random.seed(0)
sys.path.append("../../experiments")
utils = importlib.import_module("utils")

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"
plt.rcParams["axes.formatter.use_mathtext"] = True


class Toolbox:
    def __init__(
        self, model_name: str, nits: int = 50_000, episode_size: int = 500
    ) -> None:
        self.model_name = model_name
        self.nits = nits
        self.episode_size = episode_size

        self.bins_num = int(2 * nits ** (1 / 3))  # Sturges' rule
        self.stan_code: str = ""
        self.model: Any = None

        with open(f"{self.model_name}.stan", "r") as f:
            self.stan_code = f.read()
        self.model = bs.StanModel.from_stan_file(f"{self.model_name}.stan", "")

    def nuts(self) -> None:
        """
        NUTS Plot
        """
        posterior = stan.build(self.stan_code, data={}, random_seed=0)
        fit = posterior.sample(num_chains=1, num_samples=self.nits, num_warmup=20_000)

        df = fit.to_frame()
        try:
            nuts_data = df["x"].to_numpy().reshape(-1, 1)
            savemat("nuts.mat", {"nuts_data": nuts_data})

            plt.figure(figsize=(6, 6))
            plt.hist(nuts_data, bins=self.bins_num, density=True, color="#B0B1B6")
            plt.xlabel(r"$x$", fontsize=17)
            plt.ylabel(r"$\mathrm{Density}$", fontsize=17)
            plt.title("$\mathrm{NUTS}$", fontsize=17)
            plt.savefig("NUTS_demo.pdf", bbox_inches="tight", pad_inches=0)
            plt.close()

        except KeyError:
            nuts_data = df[["x.1", "x.2"]].to_numpy().reshape(-1, 2)
            savemat("nuts.mat", {"nuts_data": nuts_data})

            cm_nuts = LinearSegmentedColormap.from_list(
                "nuts_gray", [(1, 1, 1, 0), (0.69, 0.69, 0.71, 1)], N=self.bins_num
            )
            plt.figure(figsize=(6, 6))
            plt.hist2d(
                nuts_data[:, 0],
                nuts_data[:, 1],
                bins=self.bins_num,
                density=True,
                cmap=cm_nuts,
            )
            plt.xlabel(r"$x$", fontsize=17)
            plt.ylabel(r"$y$", fontsize=17)
            plt.title("$\mathrm{NUTS}$", fontsize=17)
            plt.savefig("NUTS_demo.pdf", bbox_inches="tight", pad_inches=0)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.plot(
                nuts_data[:, 0],
                nuts_data[:, 1],
                "o-",
                color="#B0B1B6",
                alpha=0.2,
                linewidth=0.5,
            )
            plt.xlabel(r"$x$", fontsize=17)
            plt.ylabel(r"$y$", fontsize=17)
            plt.title("$\mathrm{NUTS}$", fontsize=17)
            plt.savefig("NUTS_trace_demo.pdf", bbox_inches="tight", pad_inches=0)
            plt.close()

    def mala(self, mag: float = 1.0) -> None:
        """
        MALA Plot
        """
        log_p = self.model.log_density
        grad_log_p = lambda x: self.model.log_density_gradient(x)[1]
        x0 = np.zeros(self.model.param_unc_num())
        h0 = 0.1
        c0 = mag * np.eye(self.model.param_unc_num())
        alpha = 21 * [0.3]
        epoch = 20 * [1_000] + [self.nits]
        x, _, _, _, _, _, _ = utils.MALA(log_p, grad_log_p).mala_adapt(
            x0, h0, c0, alpha, epoch, pb=True
        )
        mala_data = x[-1]

        savemat("mala.mat", {"mala_data": mala_data})

        if mala_data.shape[1] == 1:
            plt.figure(figsize=(6, 6))
            plt.hist(mala_data, bins=self.bins_num, density=True, color="#91AD9E")
            plt.xlabel(r"$x$", fontsize=17)
            plt.ylabel(r"$\mathrm{Density}$", fontsize=17)
            plt.title("$\mathrm{AMALA}$", fontsize=17)
            plt.savefig("MALA_demo.pdf", bbox_inches="tight", pad_inches=0)
            plt.close()
        elif mala_data.shape[1] == 2:
            cm_mala = LinearSegmentedColormap.from_list(
                "mala_green", [(1, 1, 1, 0), (0.57, 0.68, 0.62, 1)], N=self.bins_num
            )

            plt.figure(figsize=(6, 6))
            plt.hist2d(
                mala_data[:, 0],
                mala_data[:, 1],
                bins=self.bins_num,
                density=True,
                cmap=cm_mala,
            )
            plt.xlabel(r"$x$", fontsize=17)
            plt.ylabel(r"$y$", fontsize=17)
            plt.title("$\mathrm{AMALA}$", fontsize=17)
            plt.savefig("MALA_demo.pdf", bbox_inches="tight", pad_inches=0)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.plot(
                mala_data[:, 0],
                mala_data[:, 1],
                "o-",
                color="#91AD9E",
                alpha=0.2,
                linewidth=0.5,
            )
            plt.xlabel(r"$x$", fontsize=17)
            plt.ylabel(r"$y$", fontsize=17)
            plt.title("$\mathrm{AMALA}$", fontsize=17)
            plt.savefig("MALA_trace_demo.pdf", bbox_inches="tight", pad_inches=0)
            plt.close()

        else:
            ValueError("MALA only supports 1D and 2D models.")

    def target(self, dim: int = 1, lb: float = -10, ub: float = 10) -> None:
        """
        Target Plot
        """
        match dim:
            case 1:
                x = np.linspace(lb, ub, 1000)
                gs = np.exp([self.model.log_density(np.array(i)) for i in x])

                savemat("target.mat", {"target_data": gs})

                plt.figure(figsize=(6, 6))
                plt.plot(x, gs, color="#9A7549")
                plt.xlabel(r"$x$", fontsize=17)
                plt.ylabel(r"$\mathrm{Density}$", fontsize=17)
                plt.title(r"$\mathrm{P}$", fontsize=17)
                plt.savefig("P_demo.pdf", bbox_inches="tight", pad_inches=0)
                plt.close()

            case 2:
                x = np.linspace(lb, ub, 100)
                # y = np.linspace(lb, ub, 100)
                y = np.linspace(-4, 4, 100)
                X, Y = np.meshgrid(x, y)

                Z = np.zeros_like(X)

                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        Z[i, j] = np.exp(
                            self.model.log_density(np.array([X[i, j], Y[i, j]]))
                        )

                savemat("target.mat", {"target_data": Z})

                plt.figure(figsize=(6, 6))
                plt.contour(X, Y, Z)
                plt.xlabel(r"$x$", fontsize=17)
                plt.ylabel(r"$\mathrm{Density}$", fontsize=17)
                plt.title(r"$\mathrm{P}$", fontsize=17)
                plt.savefig("P_demo.pdf", bbox_inches="tight", pad_inches=0)
                plt.close()

            case _:
                ValueError("Target only supports 1D and 2D models.")

    def rl(self) -> None:
        """
        RL-MH Plot
        """

        try:
            rl_mat_all = loadmat("train_store_accepted_sample.mat")
            rl_data_all = np.array(rl_mat_all["data"])
            rl_data = rl_data_all[-self.nits :]
        except NotImplementedError:
            rl_mat_all_t = h5py.File("train_store_accepted_sample.mat")
            rl_data_all_t = np.array(rl_mat_all_t["data"])
            rl_data_all = np.transpose(rl_data_all_t)
            rl_data = rl_data_all[-self.nits :]

        if rl_data.shape[1] == 1:
            plt.figure(figsize=(6, 6))
            plt.hist(rl_data, bins=self.bins_num, density=True, color="#686789")
            plt.xlabel(r"$x$", fontsize=17)
            plt.ylabel(r"$\mathrm{Density}$", fontsize=17)
            plt.title("$\mathrm{RLMH}$", fontsize=17)
            plt.savefig("RL_demo.pdf", bbox_inches="tight", pad_inches=0)
            plt.close()
        elif rl_data.shape[1] == 2:
            cm_rl = LinearSegmentedColormap.from_list(
                "rl_purple", [(1, 1, 1, 0), (0.29, 0.29, 0.42, 1)], N=self.bins_num
            )

            plt.figure(figsize=(6, 6))
            plt.hist2d(
                rl_data[:, 0],
                rl_data[:, 1],
                bins=self.bins_num,
                density=True,
                cmap=cm_rl,
            )
            plt.xlabel(r"$x$", fontsize=17)
            plt.ylabel(r"$y$", fontsize=17)
            plt.title("$\mathrm{RLMH}$", fontsize=17)
            plt.savefig("RL_demo.pdf", bbox_inches="tight", pad_inches=0)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.plot(
                rl_data[:, 0],
                rl_data[:, 1],
                "o-",
                color="#686789",
                alpha=0.2,
                linewidth=0.5,
            )
            plt.xlabel(r"$x$", fontsize=17)
            plt.ylabel(r"$y$", fontsize=17)
            plt.title("$\mathrm{RLMH}$", fontsize=17)
            plt.savefig("RL_trace_demo.pdf", bbox_inches="tight", pad_inches=0)
            plt.close()

    def reward(self) -> None:
        """
        Plot of Expected Squared Jump Distance with Rao Blackwell
        """
        try:
            mat_rl_reward = loadmat("train_store_reward.mat")
            data_rl_reward = np.array(mat_rl_reward["data"])
        except NotImplementedError:
            mat_rl_reward_t = h5py.File("train_store_reward.mat")
            data_rl_reward_t = np.array(mat_rl_reward_t["data"])
            data_rl_reward = np.transpose(data_rl_reward_t)

        data_rl_average_episode_reward = np.mean(
            data_rl_reward.reshape(-1, self.episode_size), axis=1
        )
        data_rl_average_episode_reward_moving_window = utils.Toolbox.moving_average(
            data_rl_average_episode_reward, 5
        )

        savemat(
            "data_rl_average_episode_reward_moving_window.mat",
            {
                "data_rl_average_episode_reward_moving_window": data_rl_average_episode_reward_moving_window
            },
        )

        step_scale = np.arange(
            0,
            self.episode_size * len(data_rl_average_episode_reward_moving_window),
            self.episode_size,
        )

        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)
        ax.plot(
            step_scale, data_rl_average_episode_reward_moving_window, color="#686789"
        )
        ax.set_xlabel(r"$n$", fontsize=17)
        ax.set_ylabel(r"$r_{n}$", fontsize=17)
        ax.ticklabel_format(style="sci", scilimits=(3, 3), axis="x")
        fig.savefig("RL_reward_demo.pdf", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
