import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['text.usetex']  = False

rng = np.random.default_rng(seed=42)

def simulate_fhn(a, t0, t_end, dt, sigma, epsilon, b, c, v0, w0):
    N = int((t_end - t0) / dt)
    times = np.linspace(t0, t_end, N+1)
    v = np.zeros(N+1)
    w = np.zeros(N+1)
    v[0], w[0] = v0, w0

    def f(v, w):
        return v * (a - v) * (v - 1) - w
    def g(v, w):
        return epsilon * (b * v - c * w)

    for n in range(N):
        dW = rng.normal(0.0, np.sqrt(dt))
        f1, g1 = f(v[n], w[n]), g(v[n], w[n])
        v[n+1] = v[n] + f1 * dt + sigma * dW
        w[n+1] = w[n] + g1 * dt
    return times, v, w

epsilon = 0.00025
a, b, c = 0.3, 1.0, 2.0
t0, t_end, dt = 0.0, 300000.0, 0.05
v0, w0 = -0.4, 0.3
sigma = 0.04

times, v, w = simulate_fhn(a, t0, t_end, dt, sigma, epsilon, b, c, v0, w0)

N_time = 20000
t_plot_start = t_end - N_time
mask = times >= t_plot_start
t_scaled = (times[mask] - t_plot_start) / N_time * 5.0

plt.figure(figsize=(8, 6))
plt.plot(t_scaled, v[mask], 'b', linewidth=2.0)
plt.plot(t_scaled, w[mask], 'r-', linewidth=2.0)
plt.xlabel(r'$t$', fontsize=32)
plt.ylabel(r'$v, w$', fontsize=32)
plt.grid(True)
ax = plt.gca()
ax.set_xlim(0, 5)
ax.set_ylim(-0.5, 1.5)
ax.set_xticks(np.arange(0, 6, 1))
y_ticks = np.linspace(-0.5, 1.5, 5)
plt.xticks(fontsize=22)
plt.yticks(y_ticks, [f"{tick:.1f}" for tick in y_ticks], fontsize=22)
ax.text(0, 1, r'(a3)', transform=ax.transAxes, fontsize=34,
        va='top', ha='left', fontweight='bold')
ax.text(0.25, 0.96, r'$\boldsymbol{a=0.3}$', transform=ax.transAxes,
        fontsize=30, va='top',
        bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', pad=0.2))
plt.tight_layout()
plt.show()

v_vals = np.linspace(-0.5, 1.2, 400)
v_null = v_vals * (a - v_vals) * (v_vals - 1)
w_null = (b / c) * v_vals

plt.plot(v_vals, v_null, 'b-', lw=2.5, label='$v$ nullcline')
plt.plot(v_vals, w_null, 'g-', lw=2.5, label='$w$ nullcline')
plt.plot(v[mask], w[mask], 'k-', lw=1.5, label='trajectory')

plt.xlabel(r'$v$', fontsize=32)
plt.ylabel(r'$w$', fontsize=32)
plt.grid(True)
ax = plt.gca()
ax.set_xlim(-0.6, 1.2)
ax.set_ylim(-0.04, 0.16)
x_ticks = np.linspace(-0.6, 1.4, 6)
y_ticks_pp = np.linspace(-0.04, 0.16, 6)
plt.xticks(x_ticks, [f"{tick:.2f}" for tick in x_ticks], fontsize=22)
plt.yticks(y_ticks_pp, [f"{tick:.2f}" for tick in y_ticks_pp], fontsize=22)
ax.text(0.77, 1.0, r'(b3)', transform=ax.transAxes, fontsize=32,
        va='top', ha='left', fontweight='bold')
plt.tight_layout()
plt.savefig('Phase_Portrait_a_3.eps', format='eps',dpi = 300, bbox_inches='tight')

plt.show()
