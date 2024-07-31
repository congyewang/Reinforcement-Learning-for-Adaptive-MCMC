import sys

sys.path.append("..")
from toolbox import Toolbox


def main():
    toolbox = Toolbox("mixture_gaussian_target")

    toolbox.nuts()
    toolbox.mala()
    toolbox.target()
    toolbox.rl()
    toolbox.reward()


if __name__ == "__main__":
    main()
