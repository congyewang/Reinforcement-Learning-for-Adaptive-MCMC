import os
import sys
import io
import re
import warnings
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

from functools import lru_cache

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"
plt.rcParams["axes.formatter.use_mathtext"] = True

# Define the type of the Extractor functions
ExtractorModel = Callable[[str, str], None]
ExtractorResult = Callable[[str, str, str, str, str], None]
ExtractorBaseline = Callable[[str, str, str, str, str, str, str], None]
ExtractorMALA = Callable[[str, str, str, str, str, str], None]
ExtractorNUTS = Callable[[str, str, str, str, str, str], None]

ExtractorMakeType = Union[
    ExtractorModel, ExtractorResult, ExtractorBaseline, ExtractorMALA, ExtractorNUTS
]


class MultiESS:
    def fminESS(
        self,
        p: int,
        alpha: Union[float, np.float64] = 0.05,
        eps: Union[float, np.float64] = 0.05,
        ess: Union[None, float, np.float64] = None,
    ) -> np.float64:
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

    def multiESS(
        self,
        X: NDArray[np.float64],
        b: str = "sqroot",
        Noffsets: int = 10,
        Nb: Union[int, None] = None,
    ) -> np.float64:
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
                "More dimensions than data points, cannot compute effective "
                "sample size."
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
        mESS = self.multiESS_chain(X, n, p, b, Noffsets, Nb)

        return mESS

    def multiESS_chain(
        self,
        Xi: NDArray[np.float64],
        n: int,
        p: int,
        b: str,
        Noffsets: int,
        Nb: Union[int, None],
    ) -> np.float64:
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
                map(
                    int, np.round(np.exp(np.linspace(np.log(b_min), np.log(b_max), Nb)))
                )
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
            mESS_i.append(self.multiESS_batch(Xi, n, p, theta, detLambda, bi, Noffsets))
        # Return lowest mESS
        mESS = np.min(mESS_i)

        return mESS

    def multiESS_batch(
        self,
        Xi: NDArray[np.float64],
        n: int,
        p: int,
        theta: NDArray[np.float64],
        detLambda: NDArray[np.float64],
        b: str,
        Noffsets: int,
    ) -> NDArray[np.float64]:
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

    def __call__(
        self,
        X: NDArray[np.float64],
        b: str = "sqroot",
        Noffsets: int = 10,
        Nb: Union[int, None] = None,
    ) -> np.float64:
        return self.multiESS(X, b=b, Noffsets=Noffsets, Nb=Nb)


multiESS = MultiESS()


class Extractor:
    def __init__(self, pdb_path: str):
        self.pdb_path = pdb_path

    def model(self, model_name: str, save_root_path: str = "trails") -> None:
        share_name: str = model_name.replace("-", "_")
        my_pdb = PosteriorDatabase(cls().pdb_path)
        posterior = my_pdb.posterior(model_name)

        # Storage Directory
        save_dir = os.path.join(save_root_path, share_name)

        # Copy Stan Code
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        shutil.copy(
            posterior.model.stan_code_file_path(),
            os.path.join(save_dir, f"{share_name}.stan"),
        )

        # Write Data Json Code
        with open(os.path.join(save_dir, f"{share_name}.json"), "w+") as f:
            f.write(json.dumps(posterior.data.values()))

        # Generate C Code
        shutil.copy(
            os.path.join("./template", "template.c"),
            os.path.join(save_dir, f"{share_name}_share.c"),
        )

        # Generate Makefile
        env = jinja2.Environment(loader=jinja2.FileSystemLoader("./template"))
        makefile_temp = env.get_template("template.Makefile.txt")
        makefile_temp_out = makefile_temp.render(model_name=share_name)
        with open(os.path.join(save_dir, f"Makefile"), "w") as f:
            f.write(makefile_temp_out)

        # Generate Head File
        shutil.copy(
            os.path.join("./template", "template.h"),
            os.path.join(save_dir, f"lib{share_name}.h"),
        )

        # Generate Bridgestan Head File
        shutil.copy(
            os.path.join("./template", "bridgestan.h"),
            os.path.join(save_dir, "bridgestan.h"),
        )

        return None

    def result(
        self,
        model_name: str,
        save_root_path: str = "results",
        config_path: str = os.path.join("template", "config.toml"),
        learning_template_path: str = os.path.join("template", "template.learning.m"),
        learning_file_name: str = "learning.m",
    ) -> None:
        share_name = model_name.replace("-", "_")

        # Load Configuration
        config_dict = toml.load(config_path)
        am_rate = config_dict[model_name]["am_rate"]

        # Storage Directory
        save_dir = os.path.join(save_root_path, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate learning.m
        template_root_path = os.path.dirname(learning_template_path)
        template_file_name = os.path.basename(learning_template_path)

        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_root_path))
        learning_matlab_temp = env.get_template(template_file_name)
        learning_matlab_temp_out = learning_matlab_temp.render(
            share_name=share_name, am_rate=am_rate
        )
        with open(os.path.join(save_dir, learning_file_name), "w") as f:
            f.write(learning_matlab_temp_out)

        return None

    def baseline(
        self,
        model_name: str,
        save_root_path: str = "baselines",
        baseline_template_path: str = os.path.join("template", "template.baseline.m"),
        baseline_file_name: str = "baseline.m",
        config_path: str = os.path.join("template", "config.toml"),
        results_root_path: str = os.path.join("results"),
        learning_file_name: str = "learning.m",
    ) -> None:
        # Load Configuration
        config_dict = toml.load(config_path)
        am_rate = config_dict[model_name]["am_rate"]

        learning_path = os.path.join(results_root_path, model_name, learning_file_name)

        with open(learning_path, "r") as f:
            learning_string = f.read()
            warm_up_rate = re.search(r"am_rate = (.+);", learning_string).group(1)
            am_rate = eval(warm_up_rate)

        share_name = model_name.replace("-", "_")

        # Storage Directory
        save_dir = os.path.join(save_root_path, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate baseline.m
        template_root_path = os.path.dirname(baseline_template_path)
        template_file_name = os.path.basename(baseline_template_path)

        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_root_path))
        baseline_matlab_temp = env.get_template(template_file_name)
        baseline_matlab_temp_out = baseline_matlab_temp.render(
            share_name=share_name, am_rate=am_rate
        )
        with open(os.path.join(save_dir, baseline_file_name), "w") as f:
            f.write(baseline_matlab_temp_out)

        return None

    def mala(
        self,
        model_name: str,
        save_root_path: str = "baselines",
        mala_template_path: str = os.path.join("template", "template.mala.py"),
        mala_file_name: str = "mala.py",
        shell_template_path: str = os.path.join("template", "template.run-mala.sh"),
        shell_script_path: str = "run-mala.sh",
    ) -> None:
        # Storage Directory
        save_dir = os.path.join(save_root_path, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate mala.py
        template_root_path = os.path.dirname(mala_template_path)
        template_file_name = os.path.basename(mala_template_path)

        if not os.path.exists(mala_template_path):
            print(f"Template file '{mala_template_path}' does not exist.")
            current_file_path = os.path.abspath(__file__)
            print(f"Current file path: {current_file_path}")

        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_root_path))
        mala_temp = env.get_template(template_file_name)
        mala_temp_out = mala_temp.render(model_name=model_name)
        with open(os.path.join(save_dir, mala_file_name), "w") as f:
            f.write(mala_temp_out)

        # Copy run-mala.sh
        shutil.copy(
            shell_template_path,
            os.path.join(save_dir, shell_script_path),
        )

        return None

    def nuts(
        self,
        model_name: str,
        save_root_path: str = "baselines",
        nuts_template_path: str = os.path.join("template", "template.nuts.py"),
        nuts_file_name: str = "nuts.py",
        shell_template_path: str = os.path.join("template", "template.run-nuts.sh"),
        shell_script_path: str = "run-nuts.sh",
    ) -> None:
        # Storage Directory
        save_dir = os.path.join(save_root_path, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate nuts.py
        template_root_path = os.path.dirname(nuts_template_path)
        template_file_name = os.path.basename(nuts_template_path)

        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_root_path))
        nuts_temp = env.get_template(template_file_name)
        nuts_temp_out = nuts_temp.render(model_name=model_name)
        with open(os.path.join(save_dir, nuts_file_name), "w") as f:
            f.write(nuts_temp_out)

        # Copy run-nuts.sh
        shutil.copy(
            os.path.join(shell_template_path),
            os.path.join(save_dir, shell_script_path),
        )

        return None

    @staticmethod
    def make(
        model_type: str,
        pdb_path: str = os.path.join("posteriordb", "posterior_database"),
    ) -> ExtractorMakeType:
        extractor = Extractor(pdb_path=pdb_path)

        match model_type:
            case "model":
                return extractor.model
            case "result":
                return extractor.result
            case "baseline":
                return extractor.baseline
            case "mala":
                return extractor.mala
            case "nuts":
                return extractor.nuts
            case _:
                raise ValueError(
                    "Invalid model type. Please choose from 'model', 'result', 'baseline', 'mala', or 'nuts'."
                )


class Toolbox:
    @staticmethod
    def flat(nested_list: List[List[Any]]) -> List[Any]:
        """
        Expand nested list
        """
        res = []
        for i in nested_list:
            if isinstance(i, list):
                res.extend(Toolbox.flat(i))
            else:
                res.append(i)
        return res

    @staticmethod
    @lru_cache(maxsize=5)
    def output_gs_name(
        dbpath: str = os.path.join("posteriordb", "posterior_database")
    ) -> List[str]:
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
                d[i] = sum(
                    Toolbox.flat(my_pdb.posterior(i).information["dimensions"].values())
                )
        df = pd.DataFrame.from_dict(d, orient="index", columns=["dimensions"])
        df.sort_values(by=["dimensions"], ascending=True, inplace=True)

        # Determining Whether the Model has a Gold Standard
        no_gs: List[str] = []
        for i in pos:
            posterior = my_pdb.posterior(i)
            try:
                posterior.reference_draws()
            except AssertionError:
                no_gs.append(i)

        # Models with a Gold Standard
        gs_models = list(set(pos).difference(set(no_gs)))

        # Delete "one_comp_mm_elim_abs-one_comp_mm_elim_abs"
        gs_models.remove("one_comp_mm_elim_abs-one_comp_mm_elim_abs")

        return gs_models

    @staticmethod
    @lru_cache(maxsize=46)
    def gold_standard(
        model_name: str, dbpath: str = os.path.join("posteriordb", "posterior_database")
    ):
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
                sum(Toolbox.flat(posterior.information["dimensions"].values())),
                posterior.reference_draws_info()["diagnostics"]["ndraws"],
            )
        )
        for i in range(len(df.keys())):
            gs_s: List[str] = []
            for j in range(len(df[df.keys()[i]])):
                gs_s += df[df.keys()[i]][j]
            gs_constrain[i] = gs_s
        gs_constrain = gs_constrain.T

        # Model Generation
        model = Toolbox.generate_model(model_name)

        gs_unconstrain = np.array(
            [model.param_unconstrain(np.array(i)) for i in gs_constrain]
        )

        return gs_unconstrain

    @staticmethod
    # @lru_cache(maxsize=46)
    def generate_model(
        model_name: str,
        dbpath: str = os.path.join("posteriordb", "posterior_database"),
        verbose: bool = False,
    ) -> Callable[[Any], Any]:
        # Load DataBase Locally
        pdb_path = os.path.join(dbpath)
        my_pdb = PosteriorDatabase(pdb_path)

        ## Load Dataset
        posterior = my_pdb.posterior(model_name)
        stan = posterior.model.stan_code_file_path()
        data = json.dumps(posterior.data.values())

        with warnings.catch_warnings():
            if verbose:
                warnings.simplefilter("default", category=UserWarning)
            else:
                warnings.simplefilter("ignore", category=UserWarning)

            model = bs.StanModel.from_stan_file(stan, data)

        return model

    @staticmethod
    @lru_cache(maxsize=5)
    def sorted_model_dict(
        dbpath: str = os.path.join("posteriordb", "posterior_database")
    ):
        model_name_list = Toolbox.output_gs_name(dbpath)
        model_sample_dim = [
            Toolbox.generate_model(i).param_unc_num() for i in model_name_list
        ]
        sorted_model_dict = dict(
            sorted(
                list(zip(model_name_list, model_sample_dim)), key=lambda x: (x[1], x[0])
            )
        )

        return sorted_model_dict

    @staticmethod
    def check_gcc_version() -> None:
        # Execute the "gcc --version" command to get the version information.
        res = subprocess.run(["gcc", "--version"], capture_output=True, text=True)

        # Check if the command was successfully executed
        if res.returncode == 0:
            # Use regular expressions to match version numbers
            match = re.search(r"(\d+\.\d+\.\d+)", res.stdout)
            if match:
                gcc_version = match.group(0)
                parsed_gcc_version = version.parse(gcc_version)

                # Check if the GCC version is 6.3.0
                if parsed_gcc_version != version.parse("6.3.0"):
                    raise ValueError("GCC version must be 6.3.0")
            else:
                raise ValueError("Unable to match version number from output.")
        else:
            raise ValueError(
                f"Unable to get GCC version information, error message.\n{res.stderr}"
            )

    @staticmethod
    def expected_square_jump_distance(data: NDArray[np.float64]) -> np.float64:
        distances = np.linalg.norm(data[1:] - data[:-1], axis=1)
        return np.mean(distances)

    @staticmethod
    def median_trick(gs: NDArray[np.float64]) -> float:
        return (0.5 * np.median(pdist(gs))).item()

    @staticmethod
    def gaussian_kernel(
        x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0
    ) -> torch.Tensor:
        """
        Compute the RBF (Gaussian) kernel between x and y.
        """

        beta = 1.0 / (2.0 * sigma**2)
        dist_sq = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-beta * dist_sq)

    @staticmethod
    def batched_mmd(
        x: torch.Tensor, y: torch.Tensor, batch_size: int = 100, sigma: float = 1.0
    ) -> torch.Tensor:
        """
        Compute the Maximum Mean Discrepancy (MMD) between x and y.
        """

        m = x.size(0)
        n = y.size(0)
        mmd_estimate_xx, mmd_estimate_yy, mmd_estimate_xy = 0.0, 0.0, 0.0

        # Compute the MMD estimate in mini-batches
        for i in range(0, m, batch_size):
            x_batch = x[i : i + batch_size]
            for j in range(0, n, batch_size):
                y_batch = y[j : j + batch_size]

                xx_kernel = Toolbox.gaussian_kernel(x_batch, x_batch, sigma)
                yy_kernel = Toolbox.gaussian_kernel(y_batch, y_batch, sigma)
                xy_kernel = Toolbox.gaussian_kernel(x_batch, y_batch, sigma)

                # Compute the MMD estimate for this mini-batch
                mmd_estimate_xx += xx_kernel.sum()
                mmd_estimate_yy += yy_kernel.sum()
                mmd_estimate_xy += xy_kernel.sum()

        # Normalize the MMD estimate
        mmd_estimate = (
            mmd_estimate_xx / m**2
            + mmd_estimate_yy / n**2
            - 2 * mmd_estimate_xy / (m * n)
        )

        return mmd_estimate

    @staticmethod
    def moving_average(
        data: NDArray[np.float64], window_size: int
    ) -> NDArray[np.float64]:
        data_flatten = data.flatten()
        window = np.ones(int(window_size)) / float(window_size)

        return np.convolve(data_flatten, window, "valid")


class Reader:
    @staticmethod
    def result(
        model_name: str,
        root_path: str = "results",
        file_name: str = "train_store_accepted_sample.mat",
    ) -> NDArray[np.float64]:
        try:
            mat = loadmat(os.path.join(root_path, model_name, file_name))
            data = np.array(mat["data"])
        except NotImplementedError:
            mat_t = h5py.File(
                os.path.join(root_path, model_name, "train_store_accepted_sample.mat")
            )
            data_t = np.array(mat_t["data"])
            data = np.transpose(data_t)

        return data

    @staticmethod
    def baseline(
        model_name: str,
        root_path: str = "baselines",
        file_name: str = "am_samples.mat",
        calculated_sample_size: int = 50_000,
    ) -> NDArray[np.float64]:
        try:
            mat_t = loadmat(os.path.join(root_path, model_name, file_name))
            data_all_t = np.array(mat_t["am_samples"])
            data_all = np.transpose(data_all_t)
        except NotImplementedError:
            mat = h5py.File(os.path.join(root_path, model_name, file_name))
            data_all = np.array(mat["am_samples"])

        data = data_all[-calculated_sample_size:]

        return data

    @staticmethod
    def nuts(
        model_name: str, root_path: str = "baselines", file_name: str = "nuts.npy"
    ) -> NDArray[np.float64]:
        data = np.load(os.path.join(root_path, model_name, file_name))

        return data

    @staticmethod
    def mala(
        model_name: str, root_path: str = "baselines", file_name: str = "mala.npy"
    ) -> NDArray[np.float64]:
        data = np.load(os.path.join(root_path, model_name, file_name))

        return data


class Evaluator:
    @staticmethod
    def result(
        model_name: str,
        results_root_path: str = "results",
        results_sample_file_name: str = "train_store_accepted_sample.mat",
        batch_size: int = 1_000,
    ) -> Tuple[int, float, float, float]:
        model = Toolbox.generate_model(model_name)
        gs_unconstrain = Toolbox.gold_standard(model_name)

        data = Reader.result(model_name, results_root_path, results_sample_file_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gs_torch = torch.from_numpy(gs_unconstrain).to(device)
        data_torch = torch.from_numpy(data).to(device)

        lengthscale = Toolbox.median_trick(gs_unconstrain)
        mmd = Toolbox.batched_mmd(
            gs_torch, data_torch, batch_size=batch_size, sigma=lengthscale
        )
        ess = multiESS(data)
        esjd = Toolbox.expected_square_jump_distance(data)

        param_unc_num = model.param_unc_num()

        return (param_unc_num, esjd.item(), ess.item(), mmd.item())

    @staticmethod
    def baseline(
        model_name: str,
        baselines_root_path: str = "baselines",
        baselines_sample_file_name: str = "am_samples.mat",
        calculated_sample_size: int = 50_000,
        batch_size: int = 1_000,
    ) -> Tuple[int, float, float, float]:
        model = Toolbox.generate_model(model_name)
        gs_unconstrain = Toolbox.gold_standard(model_name)

        data = Reader.baseline(
            model_name,
            baselines_root_path,
            baselines_sample_file_name,
            calculated_sample_size,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gs_torch = torch.from_numpy(gs_unconstrain).to(device)
        data_torch = torch.from_numpy(data).to(device)

        lengthscale = Toolbox.median_trick(gs_unconstrain)
        mmd = Toolbox.batched_mmd(
            gs_torch, data_torch, batch_size=batch_size, sigma=lengthscale
        )
        ess = multiESS(data)
        esjd = Toolbox.expected_square_jump_distance(data)

        param_unc_num = model.param_unc_num()

        return (param_unc_num, esjd.item(), ess.item(), mmd.item())

    @staticmethod
    def nuts(
        model_name: str,
        nuts_root_path: str = "baselines",
        nuts_sample_file_name: str = "nuts.npy",
        batch_size: int = 1_000,
    ) -> Tuple[int, float, float, float]:
        model = Toolbox.generate_model(model_name)
        gs_unconstrain = Toolbox.gold_standard(model_name)

        data = Reader.nuts(model_name, nuts_root_path, nuts_sample_file_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gs_torch = torch.from_numpy(gs_unconstrain).to(device)
        data_torch = torch.from_numpy(data).to(device)

        lengthscale = Toolbox.median_trick(gs_unconstrain)
        mmd = Toolbox.batched_mmd(
            gs_torch, data_torch, batch_size=batch_size, sigma=lengthscale
        )
        ess = multiESS(data)
        esjd = Toolbox.expected_square_jump_distance(data)

        param_unc_num = model.param_unc_num()

        return (param_unc_num, esjd.item(), ess.item(), mmd.item())

    @staticmethod
    def mala(
        model_name: str,
        mala_root_path: str = "baselines",
        mala_sample_file_name: str = "mala.npy",
        batch_size: int = 1_000,
    ) -> Tuple[int, float, float, float]:
        model = Toolbox.generate_model(model_name)
        gs_unconstrain = Toolbox.gold_standard(model_name)

        data = Reader.mala(model_name, mala_root_path, mala_sample_file_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gs_torch = torch.from_numpy(gs_unconstrain).to(device)
        data_torch = torch.from_numpy(data).to(device)

        lengthscale = Toolbox.median_trick(gs_unconstrain)
        mmd = Toolbox.batched_mmd(
            gs_torch, data_torch, batch_size=batch_size, sigma=lengthscale
        )
        ess = multiESS(data)
        esjd = Toolbox.expected_square_jump_distance(data)

        param_unc_num = model.param_unc_num()

        return (param_unc_num, esjd.item(), ess.item(), mmd.item())

    @staticmethod
    def make(model_type: str) -> Callable[[str], Tuple[int, float, float, float]]:
        evaluator = Evaluator()

        match model_type:
            case "results":
                return evaluator.result
            case "baselines":
                return evaluator.baseline
            case "nuts":
                return evaluator.nuts
            case "mala":
                return evaluator.mala
            case _:
                raise ValueError(
                    "Invalid model type. Please choose from 'results', 'baselines', 'nuts', or 'mala'."
                )


class Exporter:
    def __init__(
        self, root_path: str, out_root_path: str = ".", verbose: bool = True
    ) -> None:
        self.root_path = root_path
        self.out_root_path = out_root_path
        self.verbose = verbose

        self.model_type = os.path.basename(root_path)

    def export(self) -> None:
        calculate_evaluations = Evaluator.make(self.model_type)
        model_name_list: List[str] = [
            d
            for d in os.listdir(self.root_path)
            if os.path.isdir(os.path.join(self.root_path, d))
        ]
        model_table = PrettyTable()
        model_table.set_style(MARKDOWN)
        model_table.field_names = ["Task", "d", "ESJD", "ESS", "MMD"]

        for i in tqdm(model_name_list, disable=(not self.verbose)):
            param_unc_num, esjd, ess, mmd = calculate_evaluations(i)
            model_table.add_row(
                [i, param_unc_num, f"{esjd:.4e}", f"{ess:.4e}", f"{mmd:.4e}"]
            )

        output_file_path = os.path.join(
            self.out_root_path, f"{self.model_type}_table.csv"
        )

        with open(output_file_path, "w", newline="") as f:
            f.write(model_table.get_csv_string())

        return None


class PlotESJD:
    @staticmethod
    def normal(
        model_name: str,
        sample_dim: int,
        window_size: int = 5,
        episode_size: int = 500,
        save_path: str = os.path.join("pic", "esjd"),
        results_root_path: str = "results",
        baselines_root_path: str = "baselines",
        nuts_root_path: str = "baselines",
        mala_root_path: str = "baselines",
    ) -> None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Extract Data
        results_data = Reader.result(model_name, results_root_path)
        baselines_data = Reader.baseline(model_name, baselines_root_path)
        mala_data = Reader.mala(model_name, mala_root_path)
        nuts_data = Reader.nuts(model_name, nuts_root_path)

        # Calculate Expected Square Jump Distance
        (
            results_average_episode_reward_moving_window,
            baselines_average_episode_reward_moving_window,
            mala_average_episode_reward_moving_window,
            nuts_average_episode_reward_moving_window,
        ) = (
            Toolbox.moving_average(
                np.mean(
                    np.concatenate(
                        ([0.0], np.linalg.norm(i[1:] - i[:-1], axis=1))
                    ).reshape(-1, episode_size),
                    axis=1,
                ),
                window_size,
            )
            for i in [results_data, baselines_data, mala_data, nuts_data]
        )

        # Plot
        plt.figure()
        plt.plot(results_average_episode_reward_moving_window, color="#686789")
        plt.plot(baselines_average_episode_reward_moving_window, color="#B77F70")
        plt.plot(nuts_average_episode_reward_moving_window, color="#B0B1B6")
        plt.plot(mala_average_episode_reward_moving_window, color="#91AD9E")
        plt.xlabel(r"$\mathrm{Episode}$", fontsize=17)
        plt.ylabel(r"$\|x_n - x_{n+1}\|$", fontsize=17)
        plt.title(
            "$\mathrm{{{0}}} ({1}D)$".format(model_name.replace("_", "\_"), sample_dim),
            fontsize=17,
        )
        plt.savefig(os.path.join(save_path, f"{model_name}_reward.pdf"))
        plt.close()

        return None

    @staticmethod
    def compare_expected_square_jump_distance_rao_blackwell(
        model_name: str,
        sample_dim: int,
        window_size: int = 5,
        episode_size: int = 500,
        calculated_sample_size: int = 50_000,
        save_path: str = os.path.join("pic", "esjd_rb"),
        results_root_path: str = "results",
        results_file_name: str = "train_store_reward.mat",
        baselines_root_path: str = "baselines",
        baselines_file_name: str = "am_rewards.mat",
        log_scale: bool = False,
    ) -> None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        results_data = Reader.result(model_name, results_root_path, results_file_name)
        baselines_data = Reader.baseline(
            model_name, baselines_root_path, baselines_file_name, calculated_sample_size
        )

        (
            results_average_episode_reward_moving_window,
            baselines_average_episode_reward_moving_window,
        ) = (
            Toolbox.moving_average(
                np.mean(i.reshape(-1, episode_size), axis=1)
                for i in [results_data, baselines_data]
            ),
            window_size,
        )

        if log_scale:
            results_average_episode_reward_moving_window = np.log(
                results_average_episode_reward_moving_window
            )
            baselines_average_episode_reward_moving_window = np.log(
                baselines_average_episode_reward_moving_window
            )

        # Plot
        plt.figure()
        plt.plot(results_average_episode_reward_moving_window, color="#686789")
        plt.plot(baselines_average_episode_reward_moving_window, color="#B77F70")
        plt.xlabel(r"$\mathrm{Episode}$", fontsize=17)
        plt.ylabel(r"$\|x_n - x_{n+1}\|$", fontsize=17)
        plt.title(
            "$\mathrm{{{0}}} ({1}D)$".format(model_name.replace("_", "\_"), sample_dim),
            fontsize=17,
        )
        plt.savefig(os.path.join(save_path, f"{model_name}_reward_rb.pdf"))
        plt.close()

        return None


class MALA:
    @staticmethod
    def mala(
        fp: Callable[[Union[float, NDArray[np.float64]]], Union[float, np.float64]],
        fg: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        x0: NDArray[np.float64],
        h: Union[float, np.float64],
        c: NDArray[np.float64],
        n: int,
        pb: bool = True,
    ) -> Tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[Any]
    ]:
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

    @staticmethod
    def mala_adapt(
        fp: Callable[[Union[float, NDArray[np.float64]]], Union[float, np.float64]],
        fg: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        x0: NDArray[np.float64],
        h0: float,
        c0: NDArray[np.float64],
        alpha: List[float],
        epoch: List[int],
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
        x[0], g[0], p[0], a[0] = MALA.mala(fp, fg, x0, h, c, epoch[0], False)

        for i in tqdm(range(1, n_ep), disable=(not pb)):
            # Adapt preconditioning matrix
            c = alpha[i - 1] * c + (1 - alpha[i - 1]) * np.cov(x[i - 1].T)
            c = cov_nearest(c)

            # Tune step-size
            ar = np.mean(a[i - 1])
            h = h * np.exp(ar - 0.574)

            # Next epoch
            x0_new = x[i - 1][-1]
            x[i], g[i], p[i], a[i] = MALA.mala(fp, fg, x0_new, h, c, epoch[i], False)

        return (x, g, p, a, h, c)


class Sampler:
    def __init__(
        self,
        model_name: str,
        dbpath: str = os.path.join("posteriordb", "posterior_database"),
        verbose: bool = False,
    ):
        self.model_name: str = model_name
        self.dbpath = dbpath
        self.verbose = verbose

    def mala(
        self,
        epoch: List[int] = 10 * [1_000] + [50_000],
        h0: float = 0.1,
        alpha: List[float] = 11 * [0.3],
        x0: Union[None, float, np.float64, NDArray[np.float64]] = None,
    ):
        model = Toolbox.generate_model(self.model_name, self.dbpath)
        log_p: Callable[[NDArray[np.float64]], np.float64] = model.log_density
        grad_log_p: Callable[[NDArray[np.float64]], NDArray[np.float64]] = (
            lambda x: model.log_density_gradient(x)[1]
        )
        sample_dim: int = model.param_unc_num()

        if x0 is None:
            while True:
                x0 = np.random.randn(sample_dim)
                if not np.any(np.isnan(grad_log_p(x0))):
                    break

        c0 = np.eye(sample_dim)
        x, _, _, _, _, _ = MALA.mala_adapt(
            log_p, grad_log_p, x0, h0, c0, alpha, epoch, pb=self.verbose
        )

        return x[-1]

    def nuts(self):
        # Model Preparation
        ## Load DataBase Locally
        pdb_path = os.path.join(self.dbpath)
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
