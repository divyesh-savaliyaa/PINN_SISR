import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl


mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42


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


TAG = "model"
CKPT_PATH = os.path.join("checkpoints", f"{TAG}_best.pt")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
model = StatePredictor(T=ckpt["T"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
dt_model = float(ckpt.get("dt", 0.05))


def drift(X, t, a, b, c, epsilon):
    v, w = X
    dvdt = v * (a - v) * (v - 1) - w
    dwdt = epsilon * (b * v - c * w)
    return np.array([dvdt, dwdt])

def diffusion(X, t, sigma):
    return np.array([[sigma], [0.0]])

def simulate_series(a, sigma,
                    b=1.0, c=2.0, epsilon=0.00025,
                    X0=np.array([-0.4, 0.3]),
                    tmax=1_000_000, dt=0.1, seed=42):
    
    rng = np.random.default_rng(seed=seed)
    t = np.arange(0, tmax + dt, dt)


    dW = rng.normal(0, np.sqrt(dt), size=(len(t), 2))

    X = np.zeros((len(t), 2))
    X[0] = X0

    for i in range(1, len(t)):
 
        f = drift(X[i - 1], t[i - 1], a, b, c, epsilon) 
     
        g = diffusion(X[i - 1], t[i - 1], sigma).flatten()

    
        dWv   = dW[i, 0]
        Xi_v  = dWv / np.sqrt(dt)
        Eta_v = Xi_v**2 - 1.0

        v_next = X[i - 1, 0] + f[0]*dt + g[0]*dWv + 0.5 * g[0] * sigma * Eta_v * dt
        w_next = X[i - 1, 1] + f[1]*dt  

        X[i, 0] = v_next
        X[i, 1] = w_next

    v = X[:, 0]
    w = X[:, 1]
    noise_tp1 = sigma * dW[1:, 0]  
    return t, v, w, noise_tp1


def detect_spikes_dual(t, v, main_threshold=0.4, secondary_threshold=0.6, start_time=1000.0):
    spikes = []
    crossing_0_4 = False
    for i in range(1, len(v)):
        if t[i] > start_time:
            if v[i - 1] < main_threshold < v[i]:
                crossing_0_4 = True
            if crossing_0_4 and v[i] > secondary_threshold:
                spikes.append(t[i])
                crossing_0_4 = False
    return np.array(spikes)

def compute_inter_spike_intervals(spike_times):
    if len(spike_times) < 2:
        return np.array([])
    return np.diff(spike_times)

def compute_cv(isi):
    if len(isi) == 0:
        return np.nan
    mean_isi = np.mean(isi)
    if mean_isi == 0:
        return np.nan
    std_isi = np.std(isi, ddof=1)
    return std_isi / mean_isi


a = 0.05
sigma_values = np.linspace(0.0, 0.15, 50)
epsilons = [0.00025, 0.0025, 0.0075, 0.015, 0.025, 0.04]


os.makedirs("pred_series_eps", exist_ok=True)

for eps_ in epsilons:
    for sig in sigma_values:
        pred_csv = os.path.join(
            "pred_series_eps",
            f"pred_series_epsilon={eps_:0.5f}_sigma={sig:0.3f}.csv"
        )
        if os.path.exists(pred_csv):
            continue  

      
        t, v, w, noise_tp1 = simulate_series(
            a,
            sig,
            epsilon=eps_,
            dt=dt_model,
            seed=42
        )

      
        v_in = torch.from_numpy(v[:-1].astype(np.float32)).unsqueeze(1)     
        w_in = torch.from_numpy(w[:-1].astype(np.float32)).unsqueeze(1)      
        n_in = torch.from_numpy(noise_tp1.astype(np.float32)).unsqueeze(1) 

        with torch.no_grad():
            v1_pred, w1_pred, *_ = model(v_in, w_in, n_in, dt=dt_model)

        v_next_pred = v1_pred[:, 0].numpy()
        t_next      = t[1:]

     
        arr = np.c_[t_next, v_next_pred]
        np.savetxt(
            pred_csv,
            arr,
            delimiter=",",
            header="time,v_pred",
            comments="",
            fmt="%.8g"
        )


for eps_ in epsilons:
    cv_values = []
    for sig in sigma_values:
        pred_csv = os.path.join(
            "pred_series_eps",
            f"pred_series_epsilon={eps_:0.5f}_sigma={sig:0.3f}.csv"
        )
        arr = np.loadtxt(pred_csv, delimiter=",", skiprows=1)
        t_next      = arr[:, 0]
        v_next_pred = arr[:, 1]

   
        spikes = detect_spikes_dual(
            t_next, v_next_pred,
            main_threshold=0.4,
            secondary_threshold=0.6,
            start_time=1000.0
        )
        isi = compute_inter_spike_intervals(spikes)
        cv_values.append(compute_cv(isi))

 
    valid_indices = [
        i for i, cv in enumerate(cv_values)
        if not (np.isnan(cv) or cv == 0)
    ]

    sigma_valid = np.array(sigma_values)[valid_indices]
    cv_valid    = np.array(cv_values)[valid_indices]

 
    csv_path = f"cv_sigma_epsilon={eps_:0.5f}.csv"
    np.savetxt(
        csv_path,
        np.c_[sigma_valid, cv_valid],
        delimiter=",",
        fmt="%.8g"
    )


for eps_ in epsilons:
    csv_path = f"cv_sigma_epsilon={eps_:0.5f}.csv"
    sigma_valid, cv_valid = np.loadtxt(csv_path, delimiter=",", unpack=True)

    if len(cv_valid) > 0:
        idx_min = np.nanargmin(cv_valid)
        print(
            f"For epsilon = {eps_:0.5f}, minimum CV {cv_valid[idx_min]:.4f} "
            f"occurs at sigma = {sigma_valid[idx_min]:.4f}"
        )
    else:
        print(
            f"For epsilon = {eps_:0.5f}, no valid CV values to compute minimum from."
        )

    plt.plot(
        sigma_valid,
        cv_valid,
        marker='o',
        label=f"epsilon = {eps_:0.5f}"
    )

plt.xlabel(r'$\sigma$', fontsize=30)
plt.ylabel('$CV$', fontsize=30)
plt.xlim(0,0.15)
plt.ylim(0,1.4)
plt.xticks(np.linspace(0, 0.15, 4))
plt.yticks(np.linspace(0, 1.4, 8))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True)
plt.legend(loc='upper center', fontsize=16)

ax = plt.gca()
leg = ax.legend(
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    borderaxespad=0.0,
    frameon=False,
    fontsize=14
)
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))


plt.savefig('CV_Sigma_eps.eps', format='eps', dpi=300, bbox_inches='tight')

plt.show()
