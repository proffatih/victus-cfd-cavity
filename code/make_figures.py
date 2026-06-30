"""
Publication figures (vector PDF + 300 dpi PNG, colorblind-safe).

Figures:
  fig_streamlines     : streamlines at Re=100,400,1000 (3 panels)
  fig_vorticity       : vorticity contours at Re=100,400,1000 (3 panels)
  fig_centerline_u    : u(y) on x=0.5 vs Ghia (3 Re overlaid)
  fig_centerline_v    : v(x) on y=0.5 vs Ghia (3 Re overlaid)
  fig_convergence     : L2 error and psi self-convergence vs h (log-log)
  fig_re_sweep        : primary & secondary vortex strength vs Re
  fig_natural         : natural-convection isotherms+streamlines (Ra panels)
  fig_nu_ra           : Nu vs Ra with power-law fit and de Vahl Davis points
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

import sys
sys.path.insert(0, HERE)
from ghia_data import GHIA_Y, GHIA_U, GHIA_X, GHIA_V

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})
# Wong colorblind-safe palette
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
RES_LIST = [100, 400, 1000]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)
    print("  wrote", name)


def load_field(Re):
    d = np.load(os.path.join(DATA, f"field_Re{Re}.npz"))
    return d["x"], d["y"], d["psi"], d["omega"]


def fig_streamlines():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, Re in zip(axes, RES_LIST):
        x, y, psi, _ = load_field(Re)
        X, Y = np.meshgrid(x, y, indexing="ij")
        lv = np.linspace(psi.min(), 0, 16)
        lv2 = np.linspace(0, max(psi.max(), 1e-6), 6)
        ax.contour(X, Y, psi, levels=np.unique(np.concatenate([lv, lv2])),
                   colors="k", linewidths=0.6)
        ax.set_title(f"Re = {Re}")
        ax.set_xlabel("$x$"); ax.set_aspect("equal")
        if Re == RES_LIST[0]:
            ax.set_ylabel("$y$")
    fig.suptitle("Streamlines in the lid-driven cavity", y=1.02)
    save(fig, "fig_streamlines")


def fig_vorticity():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, Re in zip(axes, RES_LIST):
        x, y, _, w = load_field(Re)
        X, Y = np.meshgrid(x, y, indexing="ij")
        vmax = np.percentile(np.abs(w), 99)
        cf = ax.contourf(X, Y, np.clip(w, -vmax, vmax), levels=30,
                         cmap="RdBu_r")
        ax.set_title(f"Re = {Re}")
        ax.set_xlabel("$x$"); ax.set_aspect("equal")
        if Re == RES_LIST[0]:
            ax.set_ylabel("$y$")
        cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"$\omega$")
    fig.suptitle("Vorticity field in the lid-driven cavity", y=1.02)
    save(fig, "fig_vorticity")


def fig_centerline_u():
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for c, Re in zip(CB, RES_LIST):
        d = np.loadtxt(os.path.join(RES, f"u_centerline_Re{Re}.csv"),
                       delimiter=",", skiprows=1)
        ax.plot(d[:, 1], d[:, 0], "-", color=c, label=f"Present, Re={Re}")
        ax.plot(GHIA_U[Re], GHIA_Y, "o", color=c, mfc="none", ms=6)
    ax.plot([], [], "ko", mfc="none", label="Ghia et al. (1982)")
    ax.set_xlabel("$u$ on vertical centerline ($x=0.5$)")
    ax.set_ylabel("$y$")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "fig_centerline_u")


def fig_centerline_v():
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for c, Re in zip(CB, RES_LIST):
        d = np.loadtxt(os.path.join(RES, f"v_centerline_Re{Re}.csv"),
                       delimiter=",", skiprows=1)
        ax.plot(d[:, 0], d[:, 1], "-", color=c, label=f"Present, Re={Re}")
        ax.plot(GHIA_X, GHIA_V[Re], "s", color=c, mfc="none", ms=6)
    ax.plot([], [], "ks", mfc="none", label="Ghia et al. (1982)")
    ax.set_xlabel("$x$ on horizontal centerline ($y=0.5$)")
    ax.set_ylabel("$v$")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "fig_centerline_v")


def fig_convergence():
    mms = np.loadtxt(os.path.join(RES, "convergence_mms.csv"),
                     delimiter=",", skiprows=1)
    psi = np.loadtxt(os.path.join(RES, "convergence_psi.csv"),
                     delimiter=",", skiprows=1)
    with open(os.path.join(RES, "convergence_summary.json")) as fh:
        cs = json.load(fh)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.loglog(mms[:, 1], mms[:, 2], "o-", color=CB[0], label="MMS $L_2$ error")
    ref = mms[-1, 2] * (mms[:, 1] / mms[-1, 1])**2
    a1.loglog(mms[:, 1], ref, "k--", label="2nd-order slope")
    a1.set_xlabel("$h$"); a1.set_ylabel("$L_2$ error")
    a1.set_title(f"Poisson operator (order $\\approx${cs['mms_order_mean']:.2f})")
    a1.legend(fontsize=9)
    # self-convergence: error vs extrapolated value
    fext = cs["psi_extrapolated"]
    err = np.abs(psi[:, 2] - fext)
    a2.loglog(psi[:, 1], err, "s-", color=CB[1],
              label=r"$|\psi_{\min}-\psi_{\mathrm{ext}}|$")
    ref2 = err[-1] * (psi[:, 1] / psi[-1, 1])**2
    a2.loglog(psi[:, 1], ref2, "k--", label="2nd-order slope")
    a2.set_xlabel("$h$"); a2.set_ylabel(r"error in $\psi_{\min}$")
    a2.set_title(f"Primary vortex (order $\\approx${cs['psi_observed_order']:.2f})")
    a2.legend(fontsize=9)
    save(fig, "fig_convergence")


def fig_re_sweep():
    d = np.loadtxt(os.path.join(RES, "re_sweep.csv"), delimiter=",",
                   skiprows=1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.plot(d[:, 0], -d[:, 1], "o-", color=CB[0])
    a1.set_xlabel("Re"); a1.set_ylabel(r"$|\psi_{\min}|$ (primary vortex)")
    a1.set_title("Primary-vortex strength")
    a2.semilogy(d[:, 0], d[:, 4], "s-", color=CB[1], label="bottom-right")
    a2.semilogy(d[:, 0], d[:, 5], "^-", color=CB[2], label="bottom-left")
    a2.set_xlabel("Re"); a2.set_ylabel(r"$\psi_{\max}$ (corner vortex)")
    a2.set_title("Secondary corner-vortex strength")
    a2.legend(fontsize=9)
    save(fig, "fig_re_sweep")


def fig_natural():
    d = np.load(os.path.join(DATA, "natural_fields.npz"))
    Ra_list = [1000, 10000, 100000, 1000000]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for j, Ra in enumerate(Ra_list):
        x = d[f"Ra{Ra}_x"]; y = d[f"Ra{Ra}_y"]
        psi = d[f"Ra{Ra}_psi"]; th = d[f"Ra{Ra}_theta"]
        X, Y = np.meshgrid(x, y, indexing="ij")
        ax0 = axes[0, j]
        cf = ax0.contourf(X, Y, th, levels=20, cmap="inferno")
        ax0.set_title(f"Ra = $10^{{{int(np.log10(Ra))}}}$")
        ax0.set_aspect("equal"); ax0.set_xticks([])
        if j == 0:
            ax0.set_ylabel("isotherms\n$y$")
        cb = fig.colorbar(cf, ax=ax0, fraction=0.046, pad=0.04)
        cb.set_label(r"$\theta$")
        ax1 = axes[1, j]
        ax1.contour(X, Y, psi, levels=18, colors="k", linewidths=0.5)
        ax1.set_aspect("equal"); ax1.set_xlabel("$x$")
        if j == 0:
            ax1.set_ylabel("streamlines\n$y$")
    fig.suptitle("Natural convection: isotherms (top) and streamlines "
                 "(bottom), Pr = 0.71", y=1.0)
    save(fig, "fig_natural")


def fig_nu_ra():
    d = np.loadtxt(os.path.join(RES, "ra_sweep.csv"), delimiter=",",
                   skiprows=1)
    with open(os.path.join(RES, "parameter_summary.json")) as fh:
        ps = json.load(fh)
    C = ps["nu_correlation"]["C"]; n = ps["nu_correlation"]["n"]
    fig, ax = plt.subplots(figsize=(6.2, 5))
    Ra = d[:, 0]
    ax.loglog(Ra, d[:, 1], "o", color=CB[0], ms=9, label="Present solver")
    ax.loglog(Ra, d[:, 2], "x", color=CB[1], ms=10, mew=2,
              label="de Vahl Davis (1983)")
    rr = np.logspace(3, 6, 100)
    ax.loglog(rr, C * rr**n, "--", color=CB[2],
              label=f"fit: $Nu={C:.3f}\\,Ra^{{{n:.3f}}}$")
    ax.set_xlabel("Rayleigh number $Ra$")
    ax.set_ylabel(r"Average Nusselt number $\overline{Nu}$")
    ax.set_title("Hot-wall heat transfer, Pr = 0.71")
    ax.legend(fontsize=9)
    save(fig, "fig_nu_ra")


if __name__ == "__main__":
    fig_streamlines()
    fig_vorticity()
    fig_centerline_u()
    fig_centerline_v()
    fig_convergence()
    fig_re_sweep()
    fig_natural()
    fig_nu_ra()
    print("All figures written to", FIG)
