import sys

sys.path.append("..")
from toolbox import Toolbox


def main():
    toolbox = Toolbox("multi_modal_gaussian_target")

    toolbox.nuts()
    toolbox.mala()
    toolbox.target(-12, 13)
    toolbox.rl()
    toolbox.reward()


if __name__ == "__main__":
    main()
