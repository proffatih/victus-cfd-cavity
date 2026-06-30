"""
Verification of the lid-driven cavity solver against Ghia et al. (1982).

Runs Re = 100, 400, 1000 on a fine grid, extracts centerline u(y) and v(x),
interpolates onto the Ghia stations, reports L2 / L_inf / RMS errors, and
saves profiles (results/*.csv) and full fields (data/*.npz).
"""
import os
import json
import time
import numpy as np

from solver import CavitySolver
from ghia_data import (GHIA_Y, GHIA_U, GHIA_X, GHIA_V, GHIA_PSI_PRIMARY)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
os.makedirs(RES, exist_ok=True)
os.makedirs(DATA, exist_ok=True)


def interp_to(xq, x, f):
    return np.interp(xq, x, f)


def primary_vortex(s):
    """Locate the global stream-function extremum (primary vortex centre)."""
    psi = s.psi
    idx = np.unravel_index(np.argmin(psi), psi.shape)
    return float(psi[idx]), float(s.x[idx[0]]), float(s.y[idx[1]])


def run_case(Re, N=257, tol=1e-7):
    t0 = time.time()
    s = CavitySolver(N=N, Re=Re, mode="lid")
    s.run(tol=tol, max_iter=400000, check_every=500)
    elapsed = time.time() - t0
    u, v = s.velocities()
    mid = N // 2  # x=0.5 and y=0.5 line (N odd => exact centre)

    # u along vertical centerline x=0.5  -> u[mid, :] as function of y
    u_center = u[mid, :]
    v_center = v[:, mid]

    u_at_ghia = interp_to(GHIA_Y, s.y, u_center)
    v_at_ghia = interp_to(GHIA_X, s.x, v_center)

    eu = u_at_ghia - GHIA_U[Re]
    ev = v_at_ghia - GHIA_V[Re]
    err = {
        "u_L2": float(np.sqrt(np.mean(eu**2))),
        "u_Linf": float(np.max(np.abs(eu))),
        "v_L2": float(np.sqrt(np.mean(ev**2))),
        "v_Linf": float(np.max(np.abs(ev))),
    }

    psi_min, xc, yc = primary_vortex(s)
    gp = GHIA_PSI_PRIMARY[Re]
    err["psi_primary"] = psi_min
    err["psi_primary_ref"] = gp["psi"]
    err["psi_rel_err_pct"] = 100.0 * abs(psi_min - gp["psi"]) / abs(gp["psi"])
    err["vortex_xy"] = (xc, yc)
    err["vortex_xy_ref"] = (gp["x"], gp["y"])
    err["iters"] = s.iters
    err["time_s"] = elapsed
    err["N"] = N

    # save profiles
    np.savetxt(
        os.path.join(RES, f"u_centerline_Re{Re}.csv"),
        np.column_stack([s.y, u_center]),
        delimiter=",", header="y,u", comments="",
    )
    np.savetxt(
        os.path.join(RES, f"v_centerline_Re{Re}.csv"),
        np.column_stack([s.x, v_center]),
        delimiter=",", header="x,v", comments="",
    )
    np.savetxt(
        os.path.join(RES, f"u_at_ghia_Re{Re}.csv"),
        np.column_stack([GHIA_Y, GHIA_U[Re], u_at_ghia, eu]),
        delimiter=",", header="y,u_ghia,u_solver,error", comments="",
    )
    np.savetxt(
        os.path.join(RES, f"v_at_ghia_Re{Re}.csv"),
        np.column_stack([GHIA_X, GHIA_V[Re], v_at_ghia, ev]),
        delimiter=",", header="x,v_ghia,v_solver,error", comments="",
    )
    # save fields
    np.savez_compressed(
        os.path.join(DATA, f"field_Re{Re}.npz"),
        x=s.x, y=s.y, psi=s.psi, omega=s.omega, u=u, v=v,
    )
    print(f"Re={Re:5d} N={N} iters={s.iters} t={elapsed:6.1f}s "
          f"u_L2={err['u_L2']:.4e} v_L2={err['v_L2']:.4e} "
          f"psi_min={psi_min:.5f} (ref {gp['psi']:.5f}, "
          f"{err['psi_rel_err_pct']:.2f}%)")
    return err


if __name__ == "__main__":
    results = {}
    for Re in (100, 400, 1000):
        results[str(Re)] = run_case(Re, N=257)
    with open(os.path.join(RES, "verification_summary.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nSaved verification_summary.json")
