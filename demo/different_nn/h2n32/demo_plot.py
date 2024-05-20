import sys

sys.path.append("../..")
sys.path.append("../../../experiments")
from toolbox import Toolbox


def main():
    toolbox = Toolbox("mixture_gaussian_target")

    toolbox.rl()
    toolbox.reward()


if __name__ == "__main__":
    main()
