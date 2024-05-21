import os
import platform
import subprocess
import tarfile
from pathlib import Path

import wget
from tqdm.auto import tqdm, trange
from utils import Extractor, Toolbox


def pre_build_bridgestan() -> None:
    bridgestan_version = "2.4.1"
    bridgestan_base_dir_path = Path("./bridgestan")
    bridgestan_tar_path = (
        bridgestan_base_dir_path / f"bridgestan-{bridgestan_version}.tar.gz"
    )
    bridgestan_dir_path = bridgestan_base_dir_path / f"bridgestan-{bridgestan_version}"

    url = f"https://github.com/roualdes/bridgestan/releases/download/v{bridgestan_version}/bridgestan-{bridgestan_version}.tar.gz"
    wget.download(url, str(bridgestan_tar_path))

    with tarfile.open(bridgestan_tar_path) as tf:
        tf.extractall(bridgestan_base_dir_path)
        bridgestan_tar_path.unlink()

    os.chdir(bridgestan_dir_path / "c-example")
    with subprocess.Popen(
        ["make", "example"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as process:
        while True:
            output_line = process.stdout.readline()
            if output_line == "" and process.poll() is not None:
                break
            if output_line:
                print(output_line.strip())

        return_code = process.poll()
        if return_code == 0:
            print("Bridgestan c-example make success")
        else:
            print(f"Bridgestan c-example make failed, return_code: {return_code}")


def make_models() -> None:
    with subprocess.Popen(
        ["make"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as process:

        while True:
            output_line = process.stdout.readline()
            if output_line == "" and process.poll() is not None:
                break
            if output_line:
                print(output_line.strip())

        return_code = process.poll()
        if return_code == 0:
            print("make success")
        else:
            print(f"make failed, return_code: {return_code}")


def main():
    # Prepare
    pre_build_bridgestan()

    # Check gcc version
    if platform.system() != "Darwin":
        Toolbox.check_gcc_version()

    gs_model_name_list = Toolbox.output_gs_name()

    # Extract Models
    for i in tqdm(gs_model_name_list):
        Extractor.make("model")(i)

    ## Run make
    make_models()

    # Extract Results
    round_num = 10

    for i in range(round_num):
        if not os.path.exists(
            results_round_root := os.path.join("results", f"round{i}")
        ):
            os.makedirs(results_round_root)
        if not os.path.exists(
            baselines_round_root := os.path.join("baselines", f"round{i}")
        ):
            os.makedirs(baselines_round_root)

    for j in trange(round_num):
        for k in gs_model_name_list:
            Extractor.make("result")(k, os.path.join("results", f"round{j}"), j)
            Extractor.make("baseline")(k, os.path.join("baselines", f"round{j}"), j)
            Extractor.make("mala")(k, os.path.join("baselines", f"round{j}"), j)
            Extractor.make("nuts")(k, os.path.join("baselines", f"round{j}"), j)


if __name__ == "__main__":
    main()
