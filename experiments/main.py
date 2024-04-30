import os
from pathlib import Path
import subprocess
import tarfile
import platform
import wget
from utils import check_gcc_version, output_gs_name, extract_trails, extract_train


def pre_build_bridgestan():
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


def main():
    pre_build_bridgestan()

    if not os.path.exists("./trails"):
        os.makedirs("./trails")

    if platform.system() != "Darwin":
        check_gcc_version()

    gs_model_name_list = output_gs_name("./posteriordb/posterior_database")

    for i in gs_model_name_list:
        if i != "one_comp_mm_elim_abs-one_comp_mm_elim_abs":
            extract_trails(i)
            extract_train(i)

    # Run make
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


if __name__ == "__main__":
    main()
