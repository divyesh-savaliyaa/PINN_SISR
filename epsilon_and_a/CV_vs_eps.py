import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'   
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['text.usetex']  = False

def drift(X, t, a, b, c, epsilon):
    v, w = X
    dvdt = v * (a - v) * (v - 1) - w
    dwdt = epsilon * (b * v - c * w)
    return np.array([dvdt, dwdt])


def diffusion(X, t, sigma):
    return np.array([[sigma], [0.0]])

def simulate_and_get_spikes(sigma, epsilon,a=0.1, b=1.0, c=2.0,  
                            X0=np.array([-0.4, 0.3]), tmax=300000, dt=0.1,
                            main_threshold=0.4, secondary_threshold=0.6,
                            start_time=1000, seed=42):
   
 
    rng = np.random.default_rng(seed)
    
    
    t = np.arange(0, tmax + dt, dt)
    
    
    dW = rng.normal(0, np.sqrt(dt), size=(len(t), 2))
    
  
    X = np.zeros((len(t), 2))
    X[0] = X0
    
   
    for i in range(1, len(t)):
        Xi = dW[i] / np.sqrt(dt)  
        Eta = Xi**2 - 1           
        drift_term = drift(X[i - 1], t[i - 1], a, b, c, epsilon)
        diffusion_term = diffusion(X[i - 1], t[i - 1], sigma).flatten()
        
        X[i] = (
            X[i - 1]
            + drift_term * dt
            + diffusion_term * dW[i]
            + 0.5 * diffusion_term * np.array([sigma]) * Eta * dt
        )
    
    v = X[:, 0]

   
    spikes = []
    crossing_0_4 = False  

    for i in range(1, len(v)):
        if t[i] > start_time:  
            if v[i - 1] < main_threshold < v[i]:
                crossing_0_4 = True
            if crossing_0_4 and v[i] > secondary_threshold:
                spikes.append(t[i])  
                crossing_0_4 = False  

    return t, v, spikes

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


sigma_values = np.linspace(0.0, 0.15, 50)


epsilon_values = [0.00025, 0.0025, 0.0075, 0.015, 0.025, 0.04]



for epsilon in epsilon_values:
    csv_path = f"cv_vs_sigma_eps={epsilon:.5f}.csv"

    if os.path.exists(csv_path):
        
        sigma_valid, cv_valid = np.loadtxt(csv_path, delimiter=",", unpack=True)
    else:
       
        cv_values = []
        for sig in sigma_values:
            _, _, spike_times = simulate_and_get_spikes(sig, epsilon=epsilon, seed=42)
            isi = compute_inter_spike_intervals(spike_times)
            cv_values.append(compute_cv(isi))

        valid_indices = [i for i, cv in enumerate(cv_values) if not (np.isnan(cv) or cv == 0)]
        if np.isclose(epsilon, 0.015):
            if len(valid_indices) > 1:
                valid_indices.pop(1)
        elif np.isclose(epsilon, 0.025):
            if len(valid_indices) > 1:
                valid_indices.pop(1)
            if len(valid_indices) > 3:
                valid_indices.pop(3)
        elif np.isclose(epsilon, 0.04):
            if len(valid_indices) > 4:
                valid_indices = valid_indices[4:]
        elif np.isclose(epsilon, 0.0075):
            if len(valid_indices) > 1:
                valid_indices = valid_indices[1:]

        sigma_valid = np.array(sigma_values)[valid_indices]
        cv_valid    = np.array(cv_values)[valid_indices]

        np.savetxt(csv_path, np.c_[sigma_valid, cv_valid], delimiter=",", fmt="%.8g")

    if len(cv_valid) > 0:
        idx_min = np.nanargmin(cv_valid)
        print(f"For epsilon = {epsilon:.5f}, minimum CV {cv_valid[idx_min]:.5f} occurs at sigma = {sigma_valid[idx_min]:.5f}")
    else:
        print(f"For epsilon = {epsilon:.5f}, no valid CV values to compute minimum from.")

    plt.plot(sigma_valid, cv_valid, marker='o', label=rf'$\varepsilon$ = {epsilon:.5f}')



plt.xlabel(r'$\sigma$', fontsize=30)
plt.ylabel('$CV$', fontsize=30)
plt.xlim(0,0.15)
plt.ylim(0,1.2)
plt.xticks(np.linspace(0, 0.15, 4))
plt.yticks(np.linspace(0, 1.2, 4))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True)
ax = plt.gca()
leg = ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
                frameon=False, fontsize=14)
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.tight_layout()
plt.show()
