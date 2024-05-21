import sys

sys.path.append("..")
from toolbox import Toolbox


def main():
    toolbox = Toolbox("multi_modal_gaussian_target")

    toolbox.nuts()
    toolbox.mala()
    toolbox.target(lb=-12, ub=13)
    toolbox.rl()
    toolbox.reward()


if __name__ == "__main__":
    main()
