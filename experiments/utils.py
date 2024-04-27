import os
import re
import json
import shutil
import subprocess
import jinja2
import torch
import numpy as np
import pandas as pd
from packaging import version
from prettytable import PrettyTable, MARKDOWN

import h5py
from scipy.io import loadmat

from scipy.stats import chi2
from scipy.special import gammaln

import bridgestan as bs
from posteriordb import PosteriorDatabase

from typing import Any, List, Tuple
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


def extract_train(
    model_name: str, pdb_path: str = "posteriordb/posterior_database"
) -> None:
    share_name = model_name.replace("-", "_")

    # Calculate Mean and Covariance
    model = generate_model(model_name, pdb_path)
    gs_constrain = gold_standard(model_name)
    gs_unconstrain = np.array(
        [model.param_unconstrain(np.array(i)) for i in gs_constrain]
    )

    sample_origin = np.zeros_like(gs_unconstrain[0])
    matlab_sample_origin = "[" + ", ".join(f"{row}" for row in sample_origin) + "]"

    # Storage Directory
    destination_dir = os.path.join("./results/", model_name)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Generate learning.m
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("./template"))
    learning_matlab_temp = env.get_template("template.learning.m")
    learning_matlab_temp_out = learning_matlab_temp.render(
        initial_sample=matlab_sample_origin,
        share_name=share_name,
    )
    with open(os.path.join(destination_dir, "learning.m"), "w") as f:
        f.write(learning_matlab_temp_out)


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


def excepted_square_jump_distance(data: NDArray[np.float64]) -> NDArray[np.float64]:
    distances = np.linalg.norm(data[1:] - data[:-1], axis=1)
    return np.mean(distances)


def rbf_kernel(x, y, sigma=1.0):
    """
    Compute the RBF (Gaussian) kernel between x and y.
    """

    beta = 1.0 / (2.0 * sigma**2)
    dist_sq = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-beta * dist_sq)


def batched_mmd(x, y, batch_size=100, sigma=1.0):
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

            xx_kernel = rbf_kernel(x_batch, x_batch, sigma)
            yy_kernel = rbf_kernel(y_batch, y_batch, sigma)
            xy_kernel = rbf_kernel(x_batch, y_batch, sigma)

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

    mmd = batched_mmd(gs_torch, data_torch, batch_size=1000, sigma=1.0)
    ess = multiESS(data)
    esjd = excepted_square_jump_distance(data)

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
        model_table.add_row([i, param_unc_num, esjd, ess, mmd])

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
