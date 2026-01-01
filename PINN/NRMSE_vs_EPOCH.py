import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("checkpoints/nrmse.csv")

x = df["epoch"].to_numpy() / 1000.0

plt.plot(x, df["nrmse_data"], lw=3, label="Data", color="red")
plt.plot(x, df["nrmse_ic"], lw=3, label="Data+ic+phy1", color="orange")
plt.plot(x, df["nrmse_phy1"], lw=3, label="Data+phy2", color="blue")
plt.plot(x, df["nrmse_phy2"], lw=3, label="Data+ic+phy1+phy2", color="green")

ax = plt.gca()
ax.set_xlim(0, 10)
ax.set_ylim(0, 2)
ax.set_xticks(np.arange(0, 12, 2))
ax.set_yticks(np.arange(0, 2.5, 0.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax.set_xlabel("Epoch", fontsize=22)
ax.set_ylabel("NRMSE", fontsize=22)
ax.legend(fontsize=16)
ax.text(1.05, -0.088, r"$\times 10^{3}$", transform=ax.transAxes, fontsize=22)

plt.tight_layout()
plt.show()
