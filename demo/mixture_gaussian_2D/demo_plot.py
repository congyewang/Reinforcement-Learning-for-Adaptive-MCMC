import sys

sys.path.append("..")
from toolbox import Toolbox


def main():
    toolbox = Toolbox("mixture_gaussian_target")

    toolbox.nuts()
    toolbox.mala(mag=2.0)
    toolbox.target(dim=2)
    toolbox.rl()
    toolbox.reward()


if __name__ == "__main__":
    main()
