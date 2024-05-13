import stan
import numpy as np
import matplotlib.pyplot as plt
import bridgestan as bs
from scipy.stats import multivariate_normal
from statsmodels.stats.correlation_tools import cov_nearest
from typing import Any, Callable, List, Tuple, Union
from numpy.typing import NDArray
from tqdm.auto import tqdm
import h5py
from scipy.io import loadmat

np.random.seed(0)

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"
plt.rcParams["axes.formatter.use_mathtext"] = True


def mala(
    fp: Callable[[Union[float, NDArray[np.float64]]], Union[float, np.float64]],
    fg: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    h: Union[float, np.float64],
    c: NDArray[np.float64],
    n: int,
    pb: bool = True,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[Any]]:
    """
    Sample from a target distribution using the Metropolis-adjusted Langevin
    algorithm.

    Args:
    fp - handle to the log-density function of the target.
    fg - handle to the gradient function of the log target.
    x0 - vector of the starting values of the Markov chain.
    h  - step-size parameter.
    c  - preconditioning matrix.
    n  - number of MCMC iterations.
    pb - a progress bar is shown if set to True (default).

    Returns:
    x  - matrix of generated points.
    g  - matrix of gradients of the log target at X.
    p  - vector of log-density values of the target at X.
    a  - binary vector indicating whether a move is accepted.
    """

    # Initialise the chain
    d = len(x0)
    x = np.empty((n, d))
    g = np.empty((n, d))
    p = np.empty(n)
    a = np.zeros(n, dtype=bool)
    x[0] = x0
    g[0] = fg(x0)
    p[0] = fp(x0)

    # For each MCMC iteration
    for i in tqdm(range(1, n), disable=(not pb)):
        # Langevin proposal
        hh = h**2
        mx = x[i - 1] + hh / 2 * np.dot(c, g[i - 1])
        s = hh * c
        y = np.random.multivariate_normal(mx, s)

        # Log acceptance probability
        py = fp(y)
        gy = fg(y)
        my = y + hh / 2 * np.dot(c, gy)
        qx = multivariate_normal.logpdf(x[i - 1], my, s)
        qy = multivariate_normal.logpdf(y, mx, s)
        acc_pr = (py + qx) - (p[i - 1] + qy)

        # Accept with probability acc_pr
        if acc_pr >= 0 or np.log(np.random.uniform()) < acc_pr:
            x[i] = y
            g[i] = gy
            p[i] = py
            a[i] = True
        else:
            x[i] = x[i - 1]
            g[i] = g[i - 1]
            p[i] = p[i - 1]

    return (x, g, p, a)


def mala_adapt(
    fp: Callable[[Union[float, NDArray[np.float64]]], Union[float, np.float64]],
    fg: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    h0: float,
    c0: NDArray[np.float64],
    alpha: List[float],
    epoch: List[float],
    pb: bool = True,
) -> Tuple[
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[float],
    List[NDArray[np.float64]],
]:
    """
    Sample from a target distribution using an adaptive version of the
    Metropolis-adjusted Langevin algorithm.

    Args:
    fp    - handle to the log-density function of the target.
    fg    - handle to the gradient function of the log target.
    x0    - vector of the starting values of the Markov chain.
    h0    - initial step-size parameter.
    c0    - initial preconditioning matrix.
    alpha - vector of learning rates for preconditioning matrix.
    epoch - vector of tuning epoch lengths.
    pb    - a progress bar is shown if set to True (default).

    Returns:
    x     - list of matrices of generated points.
    g     - list of matrices of gradients of the log target at x.
    p     - list of vectors of log-density values of the target at x.
    a     - list of binary vectors indicating whether a move is accepted.
    h     - tuned step-size.
    c     - tuned preconditioning matrix.
    """

    n_ep = len(epoch)
    x = n_ep * [None]
    g = n_ep * [None]
    p = n_ep * [None]
    a = n_ep * [None]

    # First epoch
    h = h0
    c = c0
    x[0], g[0], p[0], a[0] = mala(fp, fg, x0, h, c, epoch[0], False)

    for i in tqdm(range(1, n_ep), disable=(not pb)):
        # Adapt preconditioning matrix
        c = alpha[i - 1] * c + (1 - alpha[i - 1]) * np.cov(x[i - 1].T)
        c = cov_nearest(c)

        # Tune step-size
        ar = np.mean(a[i - 1])
        h = h * np.exp(ar - 0.574)

        # Next epoch
        x0_new = x[i - 1][-1]
        x[i], g[i], p[i], a[i] = mala(fp, fg, x0_new, h, c, epoch[i], False)

    return (x, g, p, a, h, c)


def moving_average(data: NDArray[np.float64], window_size: int) -> NDArray[np.float64]:
    data_flatten = data.flatten()
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data_flatten, window, "valid")


with open("mixture_gaussian_target.stan", "r") as f:
    stan_code = f.read()
nits = 50_000
bins_num = int(2 * nits ** (1 / 3))  # Sturges' rule
model = bs.StanModel.from_stan_file("mixture_gaussian_target.stan", "")


# NUTS
posterior = stan.build(stan_code, data={}, random_seed=0)
fit = posterior.sample(num_chains=1, num_samples=nits, num_warmup=20_000)

df = fit.to_frame()  # pandas `DataFrame, requires pandas
nuts_data = df["x"].to_numpy().reshape(-1, 1)

plt.figure(figsize=(6, 6))
plt.hist(nuts_data, bins=bins_num, density=True, color="#B0B1B6")
plt.xlabel(r"$x$", fontsize=17)
plt.ylabel(r"$\mathrm{Density}$", fontsize=17)
plt.title("$\mathrm{NUTS}$", fontsize=17)
plt.savefig("NUTS_demo.pdf", bbox_inches='tight', pad_inches=0)
plt.close()


# MALA
log_p = model.log_density
grad_log_p = lambda x: model.log_density_gradient(x)[1]
x0 = np.zeros(model.param_unc_num())
h0 = 0.1
c0 = np.eye(model.param_unc_num())
alpha = 11 * [0.3]
epoch = 10 * [2_000] + [nits]
x, _, _, _, _, _ = mala_adapt(log_p, grad_log_p, x0, h0, c0, alpha, epoch, pb=True)
mala_data = x[-1].reshape(-1, 1)

plt.figure(figsize=(6, 6))
plt.hist(mala_data, bins=bins_num, density=True, color="#91AD9E")
plt.xlabel(r"$x$", fontsize=17)
plt.ylabel(r"$\mathrm{Density}$", fontsize=17)
plt.title("$\mathrm{MALA}$", fontsize=17)
plt.savefig("MALA_demo.pdf", bbox_inches='tight', pad_inches=0)
plt.close()


# P
x = np.linspace(-10, 10, 1000)
gs = np.exp([model.log_density(np.array(i)) for i in x])

plt.figure(figsize=(6, 6))
plt.plot(x, gs, color="#9A7549")
plt.xlabel(r"$x$", fontsize=17)
plt.ylabel(r"$\mathrm{Density}$", fontsize=17)
plt.title(r"$\mathrm{P}$", fontsize=17)
plt.savefig("P_demo.pdf", bbox_inches='tight', pad_inches=0)
plt.close()


# RL
try:
    rl_mat_all = loadmat("train_store_accepted_sample.mat")
    rl_data_all = np.array(rl_mat_all["data"])
    rl_data = rl_data_all[-50_000:]
except NotImplementedError:
    rl_mat_all_t = h5py.File("train_store_accepted_sample.mat")
    rl_data_all_t = np.array(rl_mat_all_t["data"])
    rl_data_all = np.transpose(rl_data_all_t)
    rl_data = rl_data_all[-50_000:]

plt.figure(figsize=(6, 6))
plt.hist(rl_data, bins=bins_num, density=True, color="#686789")
plt.xlabel(r"$x$", fontsize=17)
plt.ylabel(r"$\mathrm{Density}$", fontsize=17)
plt.title("$\mathrm{RL-MCMC}$", fontsize=17)
plt.savefig("RL_demo.pdf", bbox_inches='tight', pad_inches=0)
plt.close()

# ESJD
try:
    mat_rl_reward = loadmat("train_store_reward.mat")
    data_rl_reward = np.array(mat_rl_reward["data"])
except NotImplementedError:
    mat_rl_reward_t = h5py.File("train_store_reward.mat")
    data_rl_reward_t = np.array(mat_rl_reward_t["data"])
    data_rl_reward = np.transpose(data_rl_reward_t)

data_rl_average_episode_reward = np.mean(data_rl_reward.reshape(-1, 500), axis=1)
data_rl_average_episode_reward_moving_window = moving_average(
    data_rl_average_episode_reward, 5
)

plt.figure(figsize=(6, 6))
plt.plot(
    data_rl_average_episode_reward_moving_window,
    color="#686789"
)
plt.xlabel(r"$\mathrm{Episode}$", fontsize=17)
plt.ylabel(r"$r_{e}$", fontsize=17)
# plt.title("$\mathrm{RL-MCMC}$", fontsize=17)
plt.savefig("RL_reward_demo.pdf", bbox_inches='tight', pad_inches=0)
plt.close()
