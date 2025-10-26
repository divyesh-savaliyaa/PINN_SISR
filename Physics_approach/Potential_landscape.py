import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['text.usetex']  = False

def U(v, a, w):
    return 0.25 * v**4 - ((a + 1) * v**3) / 3 + 0.5 * a * v**2 + w * v

def U_second_deriv(v, a):
    return 3 * v**2 - 2*(a + 1) * v + a

fig, axes = plt.subplots(3, 3, figsize=(14, 10))

plt.sca(axes[0,0])
a = 0.05
w = -0.01
v = np.linspace(-4, 4, 200)
U_vals = U(v, a, w)
U_min = np.min(U_vals)
U_min_visible = U_min
plt.plot(v, U_vals, 'b', linewidth = 3.0)
plt.xlabel("$v$", fontsize = 32 )
plt.ylabel("$U(v,w,a)$", fontsize = 32)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.grid(True)
plt.xlim([-1, 2])
plt.ylim([-0.1, 0.1])
plt.xticks(np.linspace(-1, 2, 7))
plt.yticks(np.linspace(-0.1, 0.1, 5))
ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.text(0.24, 0.76, r'$\boldsymbol{w=-0.01 < w_{sym}}$' + '\n' + r'$\boldsymbol{a\:=0.05}$', transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', pad=0.2))
ax.text(1, 1, '(a)',
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        va='top',
        ha='right')

plt.sca(axes[0,1])
def U(v, a, w):
    return 0.25 * v**4 - ((a + 1) * v**3) / 3 + 0.5 * a * v**2 + w * v
a = 0.05
w = 0
v = np.linspace(-4, 4, 200)
U_vals = U(v, a, w)
U_min = np.min(U_vals)
U_min_visible = U_min
plt.plot(v, U_vals, 'b', linewidth = 3.0)
plt.xlabel("$v$", fontsize = 32 )
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.grid(True)
plt.xlim([-1, 2])
plt.ylim([-0.1, 0.1])
plt.xticks(np.linspace(-1, 2, 7))
plt.yticks(np.linspace(-0.1, 0.1, 5))
ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.text(0.355, 0.76, r'$\boldsymbol{w_{sym}=0}$' + '\n' + r'$\boldsymbol{a\quad=0.05}$', transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', pad=0.2))
ax.text(1, 1, '(b)',
        transform=ax.transAxes,
        fontsize=20, fontweight='bold',
        va='top', ha='right')

plt.sca(axes[0,2])
a = 0.05
w = 0.01
v = np.linspace(-4, 4, 200)
U_vals = U(v, a, w)
U_min = np.min(U_vals)
U_min_visible = U_min
plt.plot(v, U_vals, 'b', linewidth = 3.0)
plt.xlabel("$v$", fontsize = 32 )
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.grid(True)
plt.xlim([-1, 2])
plt.ylim([-0.1, 0.1])
plt.xticks(np.linspace(-1, 2, 7))
plt.yticks(np.linspace(-0.1, 0.1, 5))
ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.text(0.27, 0.77, r'$\boldsymbol{w=0.01 > w_{sym}}$' + '\n' + r'$\boldsymbol{a\:=0.05}$', transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', pad=0.2))
ax.text(1, 1, '(c)',
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        va='top',
        ha='right')

plt.sca(axes[1,0])
def U_second_deriv(v, a):
    return 3 * v**2 - 2*(a + 1) * v + a
a = 0.5
w = -0.01
v = np.linspace(-4, 4, 200)
U_vals = U(v, a, w)
plt.plot(v, U_vals, 'b', linewidth=3.0)
plt.xlabel("$v$", fontsize=30)
plt.ylabel("$U(v,w,a)$", fontsize=30)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.xlim([-1, 2])
plt.ylim([-0.02, 0.06])
plt.xticks(np.linspace(-1, 2, 7))
plt.yticks(np.linspace(-0.02, 0.06, 5))
ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.text(0.27, 0.77, r'$\boldsymbol{w=-0.01 < w_{sym}}$' + '\n' + r'$\boldsymbol{a\:=0.5}$',
        transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', pad=0.2))
ax.text(1, 1, '(d)',
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        va='top',
        ha='right')
coeffs = [1, -(a+1), a, w]
roots = np.roots(coeffs)
real_roots = np.sort(roots[np.isreal(roots)].real)
turning_points = []
for r in real_roots:
    value = U(r, a, w)
    curvature = U_second_deriv(r, a)
    if curvature > 0:
        turning_points.append(('min', r, value))
    else:
        turning_points.append(('max', r, value))
turning_points.sort(key=lambda tp: tp[1])
if len(turning_points) >= 3:
    A = next(tp for tp in turning_points if tp[0] == 'min')
    C = next(tp for tp in reversed(turning_points) if tp[0] == 'min')
    B = next((tp for tp in turning_points if tp[0] == 'max' and A[1] < tp[1] < C[1]), None)
    if A and B and C:
        ax.plot(A[1], A[2], 'ko', markersize=8)
        ax.plot(C[1], C[2], 'ko', markersize=8)
        offset = 0.008
        P = (2 * B[1] - 0.5*(A[1] + C[1]),
             2 * B[2] - 0.5*(A[2] + C[2]) + offset)
        t_stop = 0.95
        t_vals = np.linspace(0, t_stop, 200)
        bezier_x = (1 - t_vals)**2 * A[1] + 2 * t_vals * (1 - t_vals) * P[0] + t_vals**2 * C[1]
        bezier_y = (1 - t_vals)**2 * A[2] + 2 * t_vals * (1 - t_vals) * P[1] + t_vals**2 * C[2]
        ax.plot(bezier_x, bezier_y, color='black', linewidth=3)
        curve_end = (bezier_x[-1], bezier_y[-1])
        t = t_stop
        dx_dt = 2*(1 - t)*(P[0] - A[1]) + 2*t*(C[1] - P[0])
        dy_dt = 2*(1 - t)*(P[1] - A[2]) + 2*t*(C[2] - P[1])
        norm = np.hypot(dx_dt, dy_dt)
        if norm == 0:
            norm = 1.0
        tangent = (dx_dt / norm, dy_dt / norm)
        arrow_length = 0.04
        arrow_end = (curve_end[0] + arrow_length * tangent[0],
                     curve_end[1] + arrow_length * tangent[1])
        arrow = FancyArrowPatch(curve_end, arrow_end,
                                arrowstyle="->",
                                mutation_scale=30,
                                color='black',
                                linewidth=3)
        ax.add_patch(arrow)
        ax.hlines(y=B[2], xmin=B[1], xmax=C[1]+0.3, colors='black', linestyles='--', linewidth=2)
        ax.hlines(y=A[2], xmin=A[1], xmax=C[1]+0.3, colors='black', linestyles='--', linewidth=2)
        ax.annotate(
            '',
            xy=(B[1], B[2]),
            xytext=(B[1], A[2]),
            arrowprops=dict(arrowstyle='<->', color='black', linewidth=2)
        )
        mid_y = (B[2] + A[2]) / 2
        offset = 0.2
        ax.annotate(
            r'$\boldsymbol{\Delta U_{\ell}(w, a)}$',
            xy=(C[1] + offset, mid_y),
            xytext=(C[1] + offset, mid_y),
            textcoords='data',
            fontsize=12,
            va='center',
            ha='left'
        )

plt.sca(axes[1,1])
def U_second_deriv(v, a):
    return 3 * v**2 - 2*(a + 1) * v + a
a = 0.5
w = 0
v = np.linspace(-4, 4, 200)
U_vals = U(v, a, w)
plt.plot(v, U_vals, 'b', linewidth=3.0)
plt.xlabel("$v$", fontsize=32)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.xlim([-1, 2])
plt.ylim([-0.02, 0.06])
plt.xticks(np.linspace(-1, 2, 7))
plt.yticks(np.linspace(-0.02, 0.06, 5))
ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.text(0.36, 0.77, r'$\boldsymbol{w_{sym}=0}$' + '\n' + r'$\boldsymbol{a_{sym}=0.5}$',
        transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', pad=0.2))
coeffs = [1, -(a+1), a, w]
roots = np.roots(coeffs)
real_roots = np.sort(roots[np.isreal(roots)].real)
turning_points = []
for r in real_roots:
    U_val = U(r, a, w)
    curvature = U_second_deriv(r, a)
    if curvature > 0:
        turning_points.append(('min', r, U_val))
    else:
        turning_points.append(('max', r, U_val))
turning_points.sort(key=lambda x: x[1])
if len(turning_points) >= 3:
    A = next(tp for tp in turning_points if tp[0] == 'min')
    C = next(tp for tp in reversed(turning_points) if tp[0] == 'min')
    B = next((tp for tp in turning_points if tp[0] == 'max' and A[1] < tp[1] < C[1]), None)
    if A and B and C:
        ax.plot(A[1], A[2], 'ko', markersize=8)
        ax.plot(B[1], B[2], 'ko', markersize=8)
        ax.plot(C[1], C[2], 'ko', markersize=8)
        offset = 0.008
        P = (2 * B[1] - 0.5 * (A[1] + C[1]),
             2 * B[2] - 0.5 * (A[2] + C[2]) + offset)
        t_stop = 0.85
        t_vals = np.linspace(0, t_stop, 200)
        bezier_x = (1 - t_vals)**2 * A[1] + 2 * t_vals * (1 - t_vals) * P[0] + t_vals**2 * C[1]
        bezier_y = (1 - t_vals)**2 * A[2] + 2 * t_vals * (1 - t_vals) * P[1] + t_vals**2 * C[2]
        curve_end = (bezier_x[-1], bezier_y[-1])
        t = t_stop
        dx_dt = 2 * (1 - t) * (P[0] - A[1]) + 2 * t * (C[1] - P[0])
        dy_dt = 2 * (1 - t) * (P[1] - A[2]) + 2 * t * (C[2] - P[1])
        norm = np.hypot(dx_dt, dy_dt)
        if norm == 0:
            norm = 1.0
        tangent = (dx_dt / norm, dy_dt / norm)
        arrow_length = 0.04
        arrow_end = (curve_end[0] + arrow_length * tangent[0],
                     curve_end[1] + arrow_length * tangent[1])
        arrow_bezier = FancyArrowPatch(curve_end, arrow_end,
                                       arrowstyle="->",
                                       mutation_scale=30,
                                       color='black',
                                       linewidth=3)
        ax.hlines(y=B[2], xmin=B[1], xmax=C[1]+0.3, colors='black', linestyles='--', linewidth=2)
        ax.hlines(y=B[2], xmin=A[1]-0.32, xmax=B[1], colors='black', linestyles='--', linewidth=2)
        ax.hlines(y=A[2], xmin=A[1], xmax=C[1]+0.3, colors='black', linestyles='--', linewidth=2)
        ax.hlines(y=C[2], xmin=A[1]-0.32, xmax=C[1], colors='black', linestyles='--', linewidth=2)
        midpoint = ((A[1] + C[1]) / 2, (A[2] + C[2]) / 2)
        vertical_arrow = FancyArrowPatch((B[1], B[2]), (midpoint[0], midpoint[1]),
                                         arrowstyle='<->', mutation_scale=20, color='black', linewidth=2)
        ax.add_patch(vertical_arrow)
        arrow_mid_x = (B[1] + midpoint[0]) / 2
        arrow_mid_y = (B[2] + midpoint[1]) / 2
        text_offset = -0.03
        dashed_mid_x = (A[1] + C[1]) / 2
        dashed_mid_y = (A[2] + C[2]) / 2
        mid_y = (B[2] + A[2]) / 2
        offset = 0.2
        mid_y1 = (B[2] + C[2]) / 2
        ax.annotate(
            r'$\boldsymbol{\Delta U_{r}(w, a)}$',
            xy=(A[1]-0.96, mid_y1),
            xytext=(A[1]-0.96, mid_y1),
            textcoords='data',
            fontsize=12,
            va='center',
            ha='left'
        )
        ax.annotate(
            r'$\boldsymbol{\Delta U_{\ell}(w, a)}$',
            xy=(C[1] + offset, mid_y),
            xytext=(C[1] + offset, mid_y),
            textcoords='data',
            fontsize=12,
            va='center',
            ha='left'
        )
        ax.text(1, 0.98, '(e)',
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        va='top',
        ha='right')

plt.sca(axes[1,2])
def U_second_deriv(v, a):
    return 3 * v**2 - 2*(a + 1) * v + a
a = 0.5
w = 0.01
v = np.linspace(-4, 4, 200)
U_vals = U(v, a, w)
plt.plot(v, U_vals, 'b', linewidth=3.0)
plt.xlabel("$v$", fontsize=32)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.xlim([-1, 2])
plt.ylim([-0.02, 0.06])
plt.xticks(np.linspace(-1, 2, 7))
plt.yticks(np.linspace(-0.02, 0.06, 5))
ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.text(0.29, 0.79, r'$\boldsymbol{w=0.01 > w_{sym}}$' + '\n' + r'$\boldsymbol{a\:=0.5}$',
        transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', pad=0.2))
coeffs = [1, -(a+1), a, w]
roots = np.roots(coeffs)
real_roots = np.sort(roots[np.isreal(roots)].real)
turning_points = []
for r in real_roots:
    value = U(r, a, w)
    curvature = U_second_deriv(r, a)
    if curvature > 0:
        turning_points.append(('min', r, value))
    else:
        turning_points.append(('max', r, value))
turning_points.sort(key=lambda tp: tp[1])
if len(turning_points) >= 3:
    A = next(tp for tp in turning_points if tp[0] == 'min')
    C = next(tp for tp in reversed(turning_points) if tp[0] == 'min')
    B = next((tp for tp in turning_points if tp[0] == 'max' and A[1] < tp[1] < C[1]), None)
    if A and B and C:
        ax.plot(A[1], A[2], 'ko', markersize=8)
        ax.plot(C[1], C[2], 'ko', markersize=8)
        offset = 0.008
        P = (2 * B[1] - 0.5*(C[1] + A[1]),
             2 * B[2] - 0.5*(C[2] + A[2]) + offset)
        t_stop = 0.95
        t_vals = np.linspace(0, t_stop, 200)
        bezier_x = (1 - t_vals)**2 * C[1] + 2 * t_vals * (1 - t_vals) * P[0] + t_vals**2 * A[1]
        bezier_y = (1 - t_vals)**2 * C[2] + 2 * t_vals * (1 - t_vals) * P[1] + t_vals**2 * A[2]
        ax.plot(bezier_x, bezier_y, color='black', linewidth=3)
        curve_end = (bezier_x[-1], bezier_y[-1])
        t = t_stop
        dx_dt = 2*(1 - t)*(P[0] - C[1]) + 2*t*(A[1] - P[0])
        dy_dt = 2*(1 - t)*(P[1] - C[2]) + 2*t*(A[2] - P[1])
        norm = np.hypot(dx_dt, dy_dt)
        if norm == 0:
            norm = 1.0
        tangent = (dx_dt / norm, dy_dt / norm)
        arrow_length = 0.04
        arrow_end = (curve_end[0] + arrow_length * tangent[0],
                     curve_end[1] + arrow_length * tangent[1])
        arrow = FancyArrowPatch(curve_end, arrow_end,
                                arrowstyle="->",
                                mutation_scale=30,
                                color='black',
                                linewidth=3)
        ax.add_patch(arrow)
        ax.hlines(y=C[2], xmin=A[1]-0.32, xmax=C[1], colors='black', linestyles='--', linewidth=2)
        ax.hlines(y=B[2], xmin=A[1]-0.32, xmax=B[1], colors='black', linestyles='--', linewidth=2)
        ax.annotate(
            '',
            xy=(B[1], B[2]),
            xytext=(B[1], C[2]),
            arrowprops=dict(arrowstyle='<->', color='black', linewidth=2)
        )
        mid_y = (B[2] + C[2]) / 2
        offset = 0.02
        ax.annotate(
            r'$\boldsymbol{\Delta U_{r}(w)}$',
            xy=(A[1]-0.88 + offset, mid_y),
            xytext=(A[1]-0.88 + offset, mid_y),
            textcoords='data',
            fontsize=12,
            va='center',
            ha='left'
        )
        ax.text(1, 1, '(f)',
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        va='top',
        ha='right')

plt.sca(axes[2,0])
a = 1
w = -0.01
v = np.linspace(-4, 4, 200)
U_vals = U(v, a, w)
U_min = np.min(U_vals)
U_min_visible = U_min
plt.plot(v, U_vals, 'b', linewidth = 3.0)
plt.xlabel("$v$", fontsize = 32 )
plt.ylabel("$U(v,w,a)$", fontsize=30)

plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.grid(True)
plt.xlim([-1, 2])
plt.ylim([-0.03, 0.15])
plt.xticks(np.linspace(-1, 2, 7))
plt.yticks(np.linspace(-0.03, 0.15, 7))
ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.text(0.25, 0.76, r'$\boldsymbol{w=-0.01 < w_{sym}}$' + '\n' + r'$\boldsymbol{a\:=1}$',
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', pad=0.2))
ax.text(1, 1, '(g)',
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        va='top',
        ha='right')

plt.sca(axes[2,1])
a = 1
w = 0
v = np.linspace(-4, 4, 200)
U_vals = U(v, a, w)
U_min = np.min(U_vals)
U_min_visible = U_min
plt.plot(v, U_vals, 'b', linewidth = 3.0)
plt.xlabel("$v$", fontsize = 32 )
plt.grid(True)
plt.xlim([-1, 2])
plt.ylim([-0.03, 0.15])
plt.xticks(np.linspace(-1, 2, 7))
plt.yticks(np.linspace(-0.03, 0.15, 7))
ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.text(0.378, 0.76, r'$\boldsymbol{w_{sym}=0}$' + '\n' + r'$\boldsymbol{a\quad=1}$', transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', pad=0.2))
ax.text(1, 1, '(h)',
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        va='top',
        ha='right')

plt.sca(axes[2,2])
a = 1
w = 0.01
v = np.linspace(-4, 4, 200)
U_vals = U(v, a, w)
U_min = np.min(U_vals)
U_min_visible = U_min
plt.plot(v, U_vals, 'b', linewidth = 3.0)
plt.xlabel("$v$", fontsize = 32 )

plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.grid(True)
plt.xlim([-1, 2])
plt.ylim([-0.03, 0.15])
plt.xticks(np.linspace(-1, 2, 7))
plt.yticks(np.linspace(-0.03, 0.15, 7))
ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.text(0.25, 0.76, r'$\boldsymbol{w=0.01 > w_{sym}}$' + '\n' + r'$\boldsymbol{a\:=1}$',
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', pad=0.2))
ax.text(1, 1, '(i)',
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        va='top',
        ha='right')

plt.show()
