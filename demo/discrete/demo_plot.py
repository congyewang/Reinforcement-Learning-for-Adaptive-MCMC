import sys

sys.path.append("..")
from toolbox import Toolbox


def main():
    toolbox = Toolbox("discrete_target")

    toolbox.nuts()
    toolbox.mala()
    toolbox.target(-6, 6)
    toolbox.rl()
    toolbox.reward()


if __name__ == "__main__":
    main()
