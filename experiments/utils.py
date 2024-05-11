import os
import sys
import io
import re
import toml
import json
import shutil
import subprocess
import jinja2
import torch
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from statsmodels.stats.correlation_tools import cov_nearest
import matplotlib.pyplot as plt
from packaging import version
from prettytable import PrettyTable, MARKDOWN

import stan

import h5py
from scipy.io import loadmat

from scipy.stats import chi2
from scipy.special import gammaln
from scipy.spatial.distance import pdist

import bridgestan as bs
from posteriordb import PosteriorDatabase

from tqdm.auto import tqdm

from typing import Any, Callable, List, Tuple, Union
from numpy.typing import NDArray


def flat(nested_list: List[List[Any]]) -> List[Any]:
    """
    Expand nested list
    """
    res = []
    for i in nested_list:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res


def output_gs_name(dbpath: str = "posteriordb/posterior_database") -> List[str]:
    # Load DataBase Locally
    pdb_path = os.path.join(dbpath)
    my_pdb = PosteriorDatabase(pdb_path)

    # Extract the Names of All Models
    pos = my_pdb.posterior_names()

    # Reordering Models in Ascending Dimensional Order
    d = {}
    for i in pos:
        try:
            d[i] = sum(my_pdb.posterior(i).information["dimensions"].values())
        except TypeError:
            d[i] = sum(flat(my_pdb.posterior(i).information["dimensions"].values()))
    df = pd.DataFrame.from_dict(d, orient="index", columns=["dimensions"])
    df.sort_values(by=["dimensions"], ascending=True, inplace=True)

    # Determining Whether the Model has a Gold Standard
    no_gs = []
    for i in pos:
        posterior = my_pdb.posterior(i)
        try:
            gs = posterior.reference_draws()
        except AssertionError:
            no_gs.append(i)

    # Models with a Gold Standard
    gs_models = list(set(pos).difference(set(no_gs)))

    return gs_models


def gold_standard(model_name, dbpath="posteriordb/posterior_database"):

    # Model Preparation
    ## Load DataBase Locally
    pdb_path = os.path.join(dbpath)
    my_pdb = PosteriorDatabase(pdb_path)

    ## Load Dataset
    posterior = my_pdb.posterior(model_name)

    ## Gold Standard
    gs_list = posterior.reference_draws()
    df = pd.DataFrame(gs_list)
    gs_constrain = np.zeros(
        (
            sum(flat(posterior.information["dimensions"].values())),
            posterior.reference_draws_info()["diagnostics"]["ndraws"],
        )
    )
    for i in range(len(df.keys())):
        gs_s = []
        for j in range(len(df[df.keys()[i]])):
            gs_s += df[df.keys()[i]][j]
        gs_constrain[i] = gs_s
    gs_constrain = gs_constrain.T

    return gs_constrain


def generate_model(
    model_name: str, dbpath: str = "posteriordb/posterior_database"
) -> List[str]:
    # Load DataBase Locally
    pdb_path = os.path.join(dbpath)
    my_pdb = PosteriorDatabase(pdb_path)

    ## Load Dataset
    posterior = my_pdb.posterior(model_name)
    stan = posterior.model.stan_code_file_path()
    data = json.dumps(posterior.data.values())
    model = bs.StanModel.from_stan_file(stan, data)

    return model


def extract_trails(
    model_name: str, pdb_path: str = "posteriordb/posterior_database"
) -> None:
    share_name = model_name.replace("-", "_")
    my_pdb = PosteriorDatabase(pdb_path)
    posterior = my_pdb.posterior(model_name)

    # Storage Directory
    destination_dir = os.path.join("./trails/", share_name)

    # Copy Stan Code
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    shutil.copy(
        posterior.model.stan_code_file_path(),
        os.path.join(destination_dir, f"{share_name}.stan"),
    )

    # Write Data Json Code
    with open(os.path.join(destination_dir, f"{share_name}.json"), "w+") as f:
        f.write(json.dumps(posterior.data.values()))

    # Generate C Code
    shutil.copy(
        os.path.join("./template", "template.c"),
        os.path.join(destination_dir, f"{share_name}_share.c"),
    )

    # Generate Makefile
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("./template"))
    makefile_temp = env.get_template("template.Makefile.txt")
    makefile_temp_out = makefile_temp.render(model_name=share_name)
    with open(os.path.join(destination_dir, f"Makefile"), "w") as f:
        f.write(makefile_temp_out)

    # Generate Head File
    shutil.copy(
        os.path.join("./template", "template.h"),
        os.path.join(destination_dir, f"lib{share_name}.h"),
    )

    # Generate Bridgestan Head File
    shutil.copy(
        os.path.join("./template", "bridgestan.h"),
        os.path.join(destination_dir, "bridgestan.h"),
    )


def extract_train(model_name: str) -> None:
    share_name = model_name.replace("-", "_")

    # Load Configuration
    config_dict = toml.load(os.path.join("template", "config.toml"))
    am_rate = config_dict[model_name]["am_rate"]
    gradient_clipping = config_dict[model_name]["gc"]

    # Storage Directory
    destination_dir = os.path.join("./results/", model_name)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Generate learning.m
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("./template"))
    learning_matlab_temp = env.get_template("template.learning.m")
    learning_matlab_temp_out = learning_matlab_temp.render(
        share_name=share_name,
        am_rate=am_rate,
        gradient_clipping=(gradient_clipping * 10**6),
    )
    # learning_matlab_temp_out = learning_matlab_temp.render(
    #     share_name=share_name, am_rate=am_rate, gradient_clipping=1e-5
    # )
    with open(os.path.join(destination_dir, "learning.m"), "w") as f:
        f.write(learning_matlab_temp_out)


def extract_baseline(model_name: str) -> None:
    # Load Configuration
    config_dict = toml.load(os.path.join("template", "config.toml"))
    am_rate = config_dict[model_name]["am_rate"]

    learning_path = os.path.join("./results/", model_name, "learning.m")

    with open(learning_path, "r") as f:
        learning_string = f.read()
        warm_up_rate = re.search(r"am_rate = (.+);", learning_string).group(1)
        am_rate = eval(warm_up_rate)

    share_name = model_name.replace("-", "_")

    # Storage Directory
    destination_dir = os.path.join("./baselines/", model_name)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Generate learning.m
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("./template"))
    learning_matlab_temp = env.get_template("template.baseline.m")
    learning_matlab_temp_out = learning_matlab_temp.render(
        share_name=share_name, am_rate=am_rate
    )
    with open(os.path.join(destination_dir, "baseline.m"), "w") as f:
        f.write(learning_matlab_temp_out)


def extract_nuts(model_name: str) -> None:
    # Storage Directory
    destination_dir = os.path.join("./baselines/", model_name)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Generate nuts.py
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("./template"))
    nuts_temp = env.get_template("template.nuts.py")
    nuts_temp_out = nuts_temp.render(model_name=model_name)
    with open(os.path.join(destination_dir, "nuts.py"), "w") as f:
        f.write(nuts_temp_out)

    # Copy run-nuts.sh
    shutil.copy(
        os.path.join("./template/", "template.run-nuts.sh"),
        os.path.join(destination_dir, "run-nuts.sh"),
    )


def extract_mala(model_name: str) -> None:
    # Storage Directory
    destination_dir = os.path.join("./baselines/", model_name)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Generate mala.py
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("./template"))
    mala_temp = env.get_template("template.mala.py")
    mala_temp_out = mala_temp.render(model_name=model_name)
    with open(os.path.join(destination_dir, "mala.py"), "w") as f:
        f.write(mala_temp_out)

    # Copy run-mala.sh
    shutil.copy(
        os.path.join("./template/", "template.run-mala.sh"),
        os.path.join(destination_dir, "run-mala.sh"),
    )


def check_gcc_version():
    # Execute the "gcc --version" command to get the version information.
    res = subprocess.run(["gcc", "--version"], capture_output=True, text=True)

    # Check if the command was successfully executed
    if res.returncode == 0:
        # Use regular expressions to match version numbers
        match = re.search(r"(\d+\.\d+\.\d+)", res.stdout)
        if match:
            gcc_version = match.group(0)
            parsed_gcc_version = version.parse(gcc_version)

            assert parsed_gcc_version == version.parse("6.3.0")
        else:
            print("Unable to match version number from output.")
    else:
        print("Unable to get GCC version information, error message.")
        print(res.stderr)


def fminESS(p, alpha=0.05, eps=0.05, ess=None):
    """
    Minimum effective sample size
    """

    crit = chi2.ppf(1 - alpha, p)
    foo = 2.0 / p

    if ess is None:
        logminESS = (
            foo * np.log(2.0)
            + np.log(np.pi)
            - foo * np.log(p)
            - foo * gammaln(p / 2.0)
            - 2.0 * np.log(eps)
            + np.log(crit)
        )
        return np.round(np.exp(logminESS))
    else:
        if isinstance(ess, str):
            raise ValueError("Only numeric entry allowed for ess")
        logEPS = (
            0.5 * foo * np.log(2.0)
            + 0.5 * np.log(np.pi)
            - 0.5 * foo * np.log(p)
            - 0.5 * foo * gammaln(p / 2.0)
            - 0.5 * np.log(ess)
            + 0.5 * np.log(crit)
        )
        return np.exp(logEPS)


def multiESS(X, b="sqroot", Noffsets=10, Nb=None):
    """
    Compute multivariate effective sample size of a single Markov chain X,
    using the multivariate dependence structure of the process.

    X: MCMC samples of shape (n, p)
    n: number of samples
    p: number of parameters

    b: specifies the batch size for estimation of the covariance matrix in
       Markov chain CLT. It can take a numeric value between 1 and n/2, or a
       char value between:

    'sqroot'    b=floor(n^(1/2)) (for chains with slow mixing time; default)
    'cuberoot'  b=floor(n^(1/3)) (for chains with fast mixing time)
    'lESS'      pick the b that produces the lowest effective sample size
                for a number of b ranging from n^(1/4) to n/max(20,p); this
                is a conservative choice

    If n is not divisible by b Sigma is recomputed for up to Noffsets subsets
    of the data with different offsets, and the output mESS is the average over
    the effective sample sizes obtained for different offsets.

    Nb specifies the number of values of b to test when b='less'
    (default NB=200). This option is unused for other choices of b.

    Original source: https://github.com/lacerbi/multiESS

    Reference:
    Vats, D., Flegal, J. M., & Jones, G. L. "Multivariate Output Analysis
    for Markov chain Monte Carlo", arXiv preprint arXiv:1512.07713 (2015).

    """

    # MCMC samples and parameters
    n, p = X.shape

    if p > n:
        raise ValueError(
            "More dimensions than data points, cannot compute effective " "sample size."
        )

    # Input check for batch size B
    if isinstance(b, str):
        if b not in ["sqroot", "cuberoot", "less"]:
            raise ValueError(
                "Unknown string for batch size. Allowed arguments are "
                "'sqroot', 'cuberoot' and 'lESS'."
            )
        if b != "less" and Nb is not None:
            raise Warning(
                "Nonempty parameter NB will be ignored (NB is used "
                "only with 'lESS' batch size B)."
            )
    else:
        if not 1.0 < b < (n / 2):
            raise ValueError("The batch size B needs to be between 1 and N/2.")

    # Compute multiESS for the chain
    mESS = multiESS_chain(X, n, p, b, Noffsets, Nb)

    return mESS


def multiESS_chain(Xi, n, p, b, Noffsets, Nb):
    """
    Compute multiESS for a MCMC chain.
    """

    if b == "sqroot":
        b = [int(np.floor(n ** (1.0 / 2)))]
    elif b == "cuberoot":
        b = [int(np.floor(n ** (1.0 / 3)))]
    elif b == "less":
        b_min = np.floor(n ** (1.0 / 4))
        b_max = max(np.floor(n / max(p, 20)), np.floor(np.sqrt(n)))
        if Nb is None:
            Nb = 200
        # Try NB log-spaced values of B from B_MIN to B_MAX
        b = set(
            map(int, np.round(np.exp(np.linspace(np.log(b_min), np.log(b_max), Nb))))
        )

    # Sample mean
    theta = np.mean(Xi, axis=0)
    # Determinant of sample covariance matrix
    if p == 1:
        detLambda = np.cov(Xi.T)
    else:
        detLambda = np.linalg.det(np.cov(Xi.T))

    # Compute mESS
    mESS_i = []
    for bi in b:
        mESS_i.append(multiESS_batch(Xi, n, p, theta, detLambda, bi, Noffsets))
    # Return lowest mESS
    mESS = np.min(mESS_i)

    return mESS


def multiESS_batch(Xi, n, p, theta, detLambda, b, Noffsets):
    """
    Compute multiESS for a given batch size B.
    """

    # Compute batch estimator for SIGMA
    a = int(np.floor(n / b))
    Sigma = np.zeros((p, p))
    offsets = np.sort(
        list(set(map(int, np.round(np.linspace(0, n - np.dot(a, b), Noffsets)))))
    )

    for j in offsets:
        # Swapped a, b in reshape compared to the original code.
        Y = Xi[j + np.arange(a * b), :].reshape((a, b, p))
        Ybar = np.squeeze(np.mean(Y, axis=1))
        Z = Ybar - theta
        for i in range(a):
            if p == 1:
                Sigma += Z[i] ** 2
            else:
                Sigma += Z[i][np.newaxis, :].T * Z[i]

    Sigma = (Sigma * b) / (a - 1) / len(offsets)
    mESS = n * (detLambda / np.linalg.det(Sigma)) ** (1.0 / p)

    return mESS


def expected_square_jump_distance(data: NDArray[np.float64]) -> NDArray[np.float64]:
    distances = np.linalg.norm(data[1:] - data[:-1], axis=1)
    return np.mean(distances)


def median_trick(gs: NDArray[np.float64]) -> float:
    return (0.5 * np.median(pdist(gs))).item()


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0):
    """
    Compute the RBF (Gaussian) kernel between x and y.
    """

    beta = 1.0 / (2.0 * sigma**2)
    dist_sq = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-beta * dist_sq)


def batched_mmd(
    x: torch.Tensor, y: torch.Tensor, batch_size: int = 100, sigma: float = 1.0
):
    """
    Compute the Maximum Mean Discrepancy (MMD) between x and y.
    """

    m = x.size(0)
    n = y.size(0)
    mmd_estimate = 0.0

    # Compute the MMD estimate in mini-batches
    for i in range(0, m, batch_size):
        x_batch = x[i : i + batch_size]
        for j in range(0, n, batch_size):
            y_batch = y[j : j + batch_size]

            xx_kernel = gaussian_kernel(x_batch, x_batch, sigma)  # Median trick
            yy_kernel = gaussian_kernel(y_batch, y_batch, sigma)  # Median trick
            xy_kernel = gaussian_kernel(x_batch, y_batch, sigma)  # Median trick

            # Compute the MMD estimate for this mini-batch
            mmd_estimate += xx_kernel.mean() + yy_kernel.mean() - 2 * xy_kernel.mean()

    # Normalize the MMD estimate
    mmd_estimate /= (m // batch_size) * (n // batch_size)

    return mmd_estimate


def calculate_evaluations(
    model_name: str,
) -> Tuple[int, float, float, float]:
    model = generate_model(model_name)
    gs_constrain = gold_standard(model_name)
    gs_unconstrain = np.array(
        [model.param_unconstrain(np.array(i)) for i in gs_constrain]
    )

    try:
        mat = loadmat(f"./results/{model_name}/store_accepted_sample.mat")
        data = np.array(mat["data"])
    except NotImplementedError:
        mat_t = h5py.File(f"./results/{model_name}/store_accepted_sample.mat")
        data_t = np.array(mat_t["data"])
        data = np.transpose(data_t)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gs_torch = torch.from_numpy(gs_unconstrain).to(device)
    data_torch = torch.from_numpy(data).to(device)

    lengthscale = median_trick(gs_unconstrain)
    mmd = batched_mmd(gs_torch, data_torch, batch_size=1000, sigma=lengthscale)
    ess = multiESS(data)
    esjd = expected_square_jump_distance(data)

    param_unc_num = model.param_unc_num()

    return (
        param_unc_num,
        esjd.item(),
        ess.item(),
        mmd.item(),
    )


def calculate_evaluations_baselines(
    model_name: str,
) -> Tuple[int, float, float, float]:
    model = generate_model(model_name)
    gs_constrain = gold_standard(model_name)
    gs_unconstrain = np.array(
        [model.param_unconstrain(np.array(i)) for i in gs_constrain]
    )

    try:
        mat_t = loadmat(f"./baselines/{model_name}/am_samples.mat")
        data_all_t = np.array(mat_t["am_samples"])
        data_all = np.transpose(data_all_t)
    except NotImplementedError:
        mat = h5py.File(f"./baselines/{model_name}/am_samples.mat")
        data_all = np.array(mat["am_samples"])

    data = data_all[-50_000:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gs_torch = torch.from_numpy(gs_unconstrain).to(device)
    data_torch = torch.from_numpy(data).to(device)

    mmd = batched_mmd(gs_torch, data_torch, batch_size=1000, sigma=1.0)
    ess = multiESS(data)
    esjd = expected_square_jump_distance(data)

    param_unc_num = model.param_unc_num()

    return (
        param_unc_num,
        esjd.item(),
        ess.item(),
        mmd.item(),
    )


def export_results_table(
    results_folder_path: str = "./results",
    output_file_path: str = "./results_table.md",
    convert_to_latex: bool = False,
    convert_to_csv: bool = False,
) -> None:
    model_name = [
        d
        for d in os.listdir(results_folder_path)
        if os.path.isdir(os.path.join(results_folder_path, d))
    ]
    model_table = PrettyTable()
    model_table.set_style(MARKDOWN)
    model_table.field_names = ["Name", "Dim", "ESJD", "ESS", "MMD"]

    for i in model_name:
        param_unc_num, esjd, ess, mmd = calculate_evaluations(i)
        model_table.add_row(
            [i, param_unc_num, f"{esjd:.4e}", f"{ess:.4e}", f"{mmd:.4e}"]
        )

    with open(output_file_path, "w") as f:
        f.write(model_table.get_string(sortby="Dim", out_format="latex"))

    if convert_to_latex:
        subprocess.run(
            [
                "pandoc",
                "-s",
                output_file_path,
                "-o",
                output_file_path.replace(".md", ".tex"),
            ]
        )

    if convert_to_csv:
        with open(output_file_path.replace(".md", ".csv"), "w", newline="") as f_csv:
            f_csv.write(model_table.get_csv_string())


def export_baselines_table(
    baselines_folder_path: str = "./baselines",
    output_file_path: str = "./baselines_table.md",
    convert_to_latex: bool = False,
    convert_to_csv: bool = False,
) -> None:
    model_name = [
        d
        for d in os.listdir(baselines_folder_path)
        if os.path.isdir(os.path.join(baselines_folder_path, d))
    ]
    model_table = PrettyTable()
    model_table.set_style(MARKDOWN)
    model_table.field_names = ["Name", "Dim", "ESJD", "ESS", "MMD"]

    for i in model_name:
        param_unc_num, esjd, ess, mmd = calculate_evaluations_baselines(i)
        model_table.add_row(
            [i, param_unc_num, f"{esjd:.4e}", f"{ess:.4e}", f"{mmd:.4e}"]
        )

    with open(output_file_path, "w") as f:
        f.write(model_table.get_string(sortby="Dim", out_format="latex"))

    if convert_to_latex:
        subprocess.run(
            [
                "pandoc",
                "-s",
                output_file_path,
                "-o",
                output_file_path.replace(".md", ".tex"),
            ]
        )

    if convert_to_csv:
        with open(output_file_path.replace(".md", ".csv"), "w", newline="") as f_csv:
            f_csv.write(model_table.get_csv_string())


def compare_expected_square_jump_distance(
    model_name: str,
    window_size: int = 5,
    save_path: str = "pic/reward",
    results_base_path: str = "results",
    baselines_base_path: str = "baselines",
) -> None:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Extract results data
    try:
        mat_rl_reward = loadmat(
            os.path.join(results_base_path, model_name, "store_reward.mat")
        )
        data_rl_reward = np.array(mat_rl_reward["data"])
    except NotImplementedError:
        mat_rl_reward_t = h5py.File(
            os.path.join(results_base_path, model_name, "store_reward.mat")
        )
        data_rl_reward_t = np.array(mat_rl_reward_t["data"])
        data_rl_reward = np.transpose(data_rl_reward_t)

    # Extract baselines data
    try:
        mat_am_reward = loadmat(
            os.path.join(baselines_base_path, model_name, "am_rewards.mat")
        )
        data_am_reward = np.array(mat_am_reward["am_rewards"])
    except NotImplementedError:
        mat_am_reward_t = h5py.File(
            os.path.join(baselines_base_path, model_name, "am_rewards.mat")
        )
        data_am_reward_t = np.array(mat_am_reward_t["am_rewards"])
        data_am_reward = np.transpose(data_am_reward_t)

    data_rl_average_episode_reward = np.mean(data_rl_reward.reshape(-1, 500), axis=1)
    data_am_average_episode_reward = np.mean(
        data_am_reward[-50_000:].reshape(-1, 500), axis=1
    )

    data_rl_average_episode_reward_moving_window = moving_average(
        data_rl_average_episode_reward, window_size
    )
    data_am_average_episode_reward_moving_window = moving_average(
        data_am_average_episode_reward, window_size
    )

    # Plot
    plt.figure()
    plt.plot(data_rl_average_episode_reward_moving_window, label="RL-MCMC")
    plt.plot(data_am_average_episode_reward_moving_window, label="VA-MCMC")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"{model_name}_reward.pdf"))
    plt.close()

    return None


def moving_average(data: NDArray[np.float64], window_size: int) -> NDArray[np.float64]:
    data_flatten = data.flatten()
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data_flatten, window, "valid")


def nuts_unconstrain_samples(
    model_name: str,
    dbpath: str = "posteriordb/posterior_database",
    verbose: bool = False,
):
    # Model Preparation
    ## Load DataBase Locally
    pdb_path = os.path.join(dbpath)
    my_pdb = PosteriorDatabase(pdb_path)

    ## Load Dataset
    posterior = my_pdb.posterior(model_name)

    # Extract Stan Code and Data
    stan_code = posterior.model.stan_code()
    stan_code_path = posterior.model.stan_code_file_path()
    dict_data = posterior.data.values()
    model = bs.StanModel.from_stan_file(stan_code_path, json.dumps(dict_data))

    # Sampling
    ## Mute Stan Output
    if not verbose:
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

    nuts = stan.build(stan_code, data=dict_data)
    fit = nuts.sample(num_chains=1, num_warmup=10_000, num_samples=50_000)

    if not verbose:
        sys.stdout = stdout
        sys.stderr = stderr

    # Extract Results
    df = fit.to_frame()
    nuts_uncon = np.array(
        [
            model.param_unconstrain(np.array(i))
            for i in df[list(fit.constrained_param_names)].to_numpy()
        ]
    )

    return nuts_uncon


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


def mala_unconstrain_samples(
    model_name: str,
    dbpath: str = "posteriordb/posterior_database",
    verbose: bool = False,
):
    model = generate_model(model_name, dbpath=dbpath)
    log_p = model.log_density
    grad_log_p = lambda x: model.log_density_gradient(x)[1]

    while True:
        x0 = np.random.randn(model.param_unc_num())
        if not np.any(np.isnan(grad_log_p(x0))):
            break

    nits = 50_000
    h0 = 0.1
    c0 = np.eye(model.param_unc_num())
    alpha = 11 * [0.3]
    epoch = 10 * [1_000] + [nits]
    x, _, _, _, _, _ = mala_adapt(
        log_p, grad_log_p, x0, h0, c0, alpha, epoch, pb=verbose
    )

    return x[-1]
