import sys

sys.path.append("..")
from toolbox import Toolbox


def main():
    toolbox = Toolbox("discrete_target")

    toolbox.nuts()
    toolbox.mala(2)
    toolbox.target(lb=-7, ub=7)
    toolbox.rl()
    toolbox.reward()


if __name__ == "__main__":
    main()
