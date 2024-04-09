import os
import re
import json
import shutil
import subprocess
import jinja2
import numpy as np
import pandas as pd
from packaging import version

import bridgestan as bs
from posteriordb import PosteriorDatabase

from typing import Any, List


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
