import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt


def main(csv_path: str):
    df = pd.read_csv(csv_path)

    required = ["epoch", "share_data", "share_ic", "share_phy1", "share_phy2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df["epoch"].to_numpy() / 1000.0

    fig, ax = plt.subplots()

    ax.plot(x, df["share_data"], lw=3, label="Data", color="red")
    ax.plot(x, df["share_ic"],   lw=3, label="ic",   color="orange")
    ax.plot(x, df["share_phy1"], lw=3, label="phy1", color="blue")
    ax.plot(x, df["share_phy2"], lw=3, label="phy2", color="green")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 12, 2))
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax.set_axisbelow(True)
    ax.set_xlabel("Epoch", fontsize=22)
    ax.set_ylabel("Gradient-norm", fontsize=22)

    ax.legend(fontsize=16)

    ax.text(1.05, -0.088, r"$\times 10^{3}$", transform=ax.transAxes, fontsize=22)
    ax.text(0.5, 1.0, r"(b)", transform=ax.transAxes,
            ha="center", va="top", fontsize=36, fontweight="bold")

    fig.savefig("Grad-norm.eps", format="eps", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "grad_shares.csv"
    main(csv_path)
