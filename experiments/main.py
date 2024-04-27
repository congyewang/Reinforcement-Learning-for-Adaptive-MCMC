import os
import platform
import subprocess
from utils import check_gcc_version, output_gs_name, extract_trails, extract_train


def main():
    if not os.path.exists("./trails"):
        os.makedirs("./trails")

    if platform.system() != "Darwin":
        check_gcc_version()

    gs_model_name_list = output_gs_name("./posteriordb/posterior_database")

    for i in gs_model_name_list:
        extract_trails(i)
        extract_train(i)

    # Run make
    process = subprocess.Popen(
        ["make"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

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
