import matplotlib
import matplotlib.pyplot as plt

from .load import load_demo

matplotlib.use("Agg")


def main():
    raw = load_demo()
    raw.plot_psd(average=True)
    plt.savefig("psd.png")


if __name__ == "__main__":
    main()
