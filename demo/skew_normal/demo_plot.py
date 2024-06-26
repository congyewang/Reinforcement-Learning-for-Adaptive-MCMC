import sys

sys.path.append("..")
from toolbox import Toolbox


def main():
    toolbox = Toolbox("skew_normal_target")

    toolbox.nuts()
    toolbox.mala()
    toolbox.target(lb=-2, ub=4)
    toolbox.rl()
    toolbox.reward()


if __name__ == "__main__":
    main()
