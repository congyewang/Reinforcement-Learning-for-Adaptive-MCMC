import sys

sys.path.append("..")
from toolbox import Toolbox


def main():
    toolbox = Toolbox("mixture_gaussian_target")

    toolbox.nuts()
    toolbox.mala(mag=2)
    toolbox.target(dim=1, lb=-10, ub=15)
    toolbox.rl()
    toolbox.reward()


if __name__ == "__main__":
    main()
