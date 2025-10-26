import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42


#1) Data only
TAG = "nasp_data_only_vardt"
METRICS_CSV = Path("checkpoints") / f"{TAG}_metrics.csv"
df = pd.read_csv(METRICS_CSV, index_col=0).sort_index()
df['rollout_smooth'] = df['rollout_nrmse'].ewm(alpha=0.002, adjust=False).mean()
x = df.index.to_numpy() / 1000.0
plt.plot(x, df['rollout_smooth'], lw=3, label='Data', color='red')

#2) Data+ic+phy1
TAG = "nasp_three_loss_phy1"
METRICS_CSV = Path("checkpoints") / f"{TAG}_metrics.csv"
df = pd.read_csv(METRICS_CSV, index_col=0).sort_index()
df['rollout_smooth'] = df['rollout_nrmse'].ewm(alpha=0.005, adjust=False).mean()
x = df.index.to_numpy() / 1000.0
plt.plot(x, df['rollout_smooth'], lw=3, label='Data+ic+phy1', color='orange')

#3) Data+phy2
TAG = "nasp_loss_phy2"
METRICS_CSV = Path("checkpoints") / f"{TAG}_metrics.csv"
K = 50
df = pd.read_csv(METRICS_CSV, index_col=0).sort_index()
if K > 0:
    df = df.iloc[:-K].copy()
df['rollout_smooth'] = df['rollout_nrmse'].ewm(alpha=0.007, adjust=False).mean()
x = df.index.to_numpy() / 1000.0
plt.plot(x, df['rollout_smooth'], lw=3, label='Data+phy2', color='blue')

#4) Data+ic+phy1+phy2
TAG = "model"
METRICS_CSV = Path("checkpoints") / f"{TAG}_metrics.csv"
df = pd.read_csv(METRICS_CSV, index_col=0).sort_index()
df['rollout_smooth'] = df['rollout_nrmse'].ewm(alpha=0.005, adjust=False).mean()
x = df.index.to_numpy() / 1000.0
plt.plot(x, df['rollout_smooth'], lw=3, label='Data+ic+phy1+phy2', color='green')


x = df.index.to_numpy() / 1000.0


ax1 = plt.gca()
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 2)
ax1.set_xticks(np.arange(0, 12, 2))
ax1.set_yticks(np.arange(0, 2.5, 0.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('NRMSE'); ax1.legend(fontsize=16)
ax1.set_xlabel('Epoch', fontsize=22); ax1.set_ylabel('NRMSE', fontsize=22); ax1.legend(fontsize=16)
ax1.text(1.05, -0.088, r'$\times 10^{3}$', transform=ax1.transAxes, fontsize=22)
plt.tight_layout()
plt.savefig('Nrmse.eps', format='eps',dpi = 300, bbox_inches='tight')
plt.show()
