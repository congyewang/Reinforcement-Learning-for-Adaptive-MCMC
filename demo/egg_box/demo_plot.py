import sys

sys.path.append("..")
from toolbox import Toolbox


def main():
    toolbox = Toolbox("egg_box_target")

    toolbox.nuts()
    toolbox.mala()
    toolbox.target(dim=2)
    toolbox.rl()
    toolbox.reward()


if __name__ == "__main__":
    main()
