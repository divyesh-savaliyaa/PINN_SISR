import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'  
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['text.usetex']  = False

a = 0.5


def U(v, w, a):
    return 0.25 * v**4 - (a+1)/3 * v**3 + 0.5 * a * v**2 + w * v


w_values = np.linspace(-0.06, 0.06, 2000)

delta_U_first = []  
delta_U_second = [] 

for w in w_values:
    coeffs = [1, -(a+1), a, w]  
    roots = np.roots(coeffs)
    real_roots = [np.real(r) for r in roots if np.abs(np.imag(r)) < 1e-6]
    
    if len(real_roots) != 3:
        delta_U_first.append(np.nan)
        delta_U_second.append(np.nan)
        continue
    
    
    real_roots.sort()
    v_small = real_roots[0]
    v_mid = real_roots[1]
    v_large = real_roots[2]
    
    
    U_small = U(v_small, w, a)
    U_mid = U(v_mid, w, a)
    U_large = U(v_large, w, a)
    
 
    delta_U_first.append(U_mid - U_small)
   
    delta_U_second.append(U_mid - U_large)

delta_U_first = np.array(delta_U_first)
delta_U_second = np.array(delta_U_second)


valid = np.isfinite(delta_U_first) & np.isfinite(delta_U_second)

print(f"Scanned w range: [{w_values[0]:.6f}, {w_values[-1]:.6f}]")
if np.any(valid):
    w_valid = w_values[valid]
    print(f"w range with 3 real roots: [{w_valid.min():.6f}, {w_valid.max():.6f}]")
else:
    print("No w values with 3 real roots found in the scanned range.")

if np.any(valid):
    for name, d in [("ΔU_mid - U_small", delta_U_first),
                    ("ΔU_mid - U_large", delta_U_second)]:
        d = d.copy()
        d[~valid] = np.nan
        i_min = np.nanargmin(d); i_max = np.nanargmax(d)
        print(f"{name}: min {d[i_min]:.6f} at w = {w_values[i_min]:.6f}; "
              f"max {d[i_max]:.6f} at w = {w_values[i_max]:.6f}")
    
plt.plot(w_values, delta_U_first, 'b-', linewidth=3)
plt.plot(w_values, delta_U_second, 'r-', linewidth=3)
plt.ylabel(r'$\Delta U_{\ell, r}(w,a)$', fontsize=30)
plt.xlabel(r'$w$', fontsize=30)
plt.xlim(-0.06, 0.06)
xticks = np.linspace(-0.06, 0.06, 5)
plt.xticks(xticks, fontsize=24)
plt.ylim(-0.01, 0.05)
yticks = np.linspace(-0.01, 0.05, 4)
plt.yticks(yticks, fontsize=24)
ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.grid(True)
plt.savefig('U_W.eps', format='eps',dpi = 300, bbox_inches='tight')
plt.show()
