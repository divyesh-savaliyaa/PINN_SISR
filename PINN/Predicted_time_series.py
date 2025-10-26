import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import matplotlib.transforms as mtransforms

mpl.rcParams['mathtext.fontset'] = 'stix'   
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['text.usetex']  = False

TAG = "model"
METRICS_CSV = Path("checkpoints") / f"{TAG}_metrics.csv"
CKPT_PATH   = Path("checkpoints") / f"{TAG}_best.pt"
TRAIN_CSV   = Path("data/train_data.csv")
FULL_CSV    = Path("data/test_data.csv")
EPS_SCALE   = 2.5e-4
SPAN        = 25.0


def load_data(csv_file):
    d = pd.read_csv(csv_file)
    t = d['time'].astype(np.float32).values
    v = d['v'].astype(np.float32).values[:, None]
    w = d['w'].astype(np.float32).values[:, None]
    n = d['noise'].astype(np.float32).values[:, None]
    return t, v, w, n

def annotate_regions(ax, t_eps, t_boundary_eps, y_frac=0.83):
    ax.axvline(t_boundary_eps, color='green', lw=2, zorder=3)
    x_left  = (t_eps[0] + t_boundary_eps) / 2.0
    x_right = (t_boundary_eps + t_eps[-1]) / 2.0
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(x_left,  y_frac, 'Training phase',  ha='center', va='top',
            transform=trans, clip_on=False, fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), zorder=5)
    ax.text(x_right, y_frac, 'Prediction phase', ha='center', va='top',
            transform=trans, clip_on=False, fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), zorder=5)

class StatePredictor(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(3,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
        )
        self.dyn        = nn.Linear(128, 2)
        self.state_head = nn.Linear(128, 2)
        self.index_head = nn.Linear(3, 2)
        self.T = T
    def forward(self, v0, w0, noise, dt=0.05):
        x = torch.cat([v0, w0, noise], dim=1)
        h = self.body(x)
        dv_dw = self.dyn(h)
        dv, dw = dv_dw[:, :1], dv_dw[:, 1:]
        v1, w1 = v0 + dv*dt, w0 + dw*dt
        vhat0, what0 = self.state_head(h).chunk(2, dim=1)
        idx = torch.sigmoid(self.index_head(x)) * (self.T - 1)
        return v1, w1, vhat0, what0, idx

ckpt = torch.load(CKPT_PATH, map_location="cpu")
model = StatePredictor(T=ckpt["T"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
dt = float(ckpt.get("dt", 0.05))

t_full, v_np, w_np, n_np = load_data(FULL_CSV)
t_train, *_ = load_data(TRAIN_CSV)
t_boundary = float(t_train[-1])

v_in_np, w_in_np, n_in_np = v_np[:-1], w_np[:-1], n_np[:-1]
t_next = t_full[1:]
t_eps = EPS_SCALE * t_next
t_boundary_eps = EPS_SCALE * (t_boundary + dt)

v_in = torch.from_numpy(v_in_np)
w_in = torch.from_numpy(w_in_np)
n_in = torch.from_numpy(n_in_np)

with torch.no_grad():
    v1_pred, w1_pred, *_ = model(v_in, w_in, n_in, dt)

v_next_true = v_np[1:].flatten()
w_next_true = w_np[1:].flatten()
v_next_pred = v1_pred[:, 0:1].numpy().flatten()
w_next_pred = w1_pred[:, 0:1].numpy().flatten()

x0_raw, x1_raw = float(t_eps[0]), float(t_eps[-1])
span = (x1_raw - x0_raw) if (x1_raw > x0_raw) else 1.0
t_scaled = (t_eps - x0_raw) / span * SPAN
boundary_scaled = (t_boundary_eps - x0_raw) / span * SPAN

fig, axs = plt.subplots(1, 2, figsize=(12,5))

ax = axs[0]
ax.plot(t_scaled, v_next_true, 'b', linewidth=2.0, label='True v')
ax.plot(t_scaled, v_next_pred, 'r--', linewidth=2.0, label='Pred v')
ax.set_xlabel(r'$t$', fontsize=32)
ax.set_ylabel(r'$v$', fontsize=32)
ax.grid(True)
ax.set_xlim(0, SPAN)
ax.set_xticks(np.arange(0, SPAN + 1, 5))
ax.set_ylim(-0.5, 1.5)
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
ax.legend(loc='upper right')
annotate_regions(ax, t_scaled, boundary_scaled, y_frac=0.83)
ax.text(1.05, -0.088, r'$\times 10^{3}$', transform=ax.transAxes, fontsize=22)

ax = axs[1]
ax.plot(t_scaled, w_next_true, 'b', linewidth=2.0, label='True w')
ax.plot(t_scaled, w_next_pred, 'r--', linewidth=2.0, label='Pred w')
ax.set_xlabel(r'$t$', fontsize=32)
ax.set_ylabel(r'$w$', fontsize=32)
ax.grid(True)
ax.set_xlim(0, SPAN)
ax.set_xticks(np.arange(0, SPAN + 1, 5))
ax.set_ylim(-0.05, 0.2)
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
ax.legend(loc='upper right')
ax.text(1.05, -0.088, r'$\times 10^{3}$', transform=ax.transAxes, fontsize=22)
annotate_regions(ax, t_scaled, boundary_scaled, y_frac=0.83)

plt.tight_layout()
fig.savefig('Data3_vw.eps', format='eps', dpi=300, bbox_inches='tight')
plt.show()
