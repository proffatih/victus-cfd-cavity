"""
Spatial convergence study for the lid-driven cavity solver.

Two complementary measures at Re = 100:
  (1) A manufactured-solution (MMS) verification of the Poisson/Laplacian
      discretisation: an exact biharmonic-type field is imposed and the
      discrete operator error is measured -> confirms 2nd-order Laplacian.
  (2) A self-convergence (Richardson) study of a physical functional --
      the primary-vortex stream-function value psi_min -- across a sequence
      of refined grids, yielding the observed order of accuracy p.

Outputs: results/convergence_mms.csv, results/convergence_psi.csv,
         results/convergence_summary.json
"""
import os
import json
import numpy as np

from solver import CavitySolver, _poisson_dst

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RES, exist_ok=True)


def mms_laplacian_order():
    """Impose psi(x,y)=sin(pi x) sin(pi y); lap = -2 pi^2 psi.
    Solve Poisson with the DST solver and measure L2 error vs grid."""
    rows = []
    for N in (17, 33, 65, 129, 257):
        h = 1.0 / (N - 1)
        x = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, x, indexing="ij")
        psi_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
        rhs = -2.0 * np.pi**2 * psi_exact  # = lap(psi_exact)
        psi_num = np.zeros((N, N))
        psi_num[1:-1, 1:-1] = _poisson_dst(rhs[1:-1, 1:-1], h)
        err = np.sqrt(np.mean((psi_num - psi_exact)[1:-1, 1:-1] ** 2))
        rows.append((N, h, err))
    rows = np.array(rows)
    # observed order from successive halvings of h
    p = np.log(rows[:-1, 2] / rows[1:, 2]) / np.log(rows[:-1, 1] / rows[1:, 1])
    return rows, p


def psi_self_convergence(Re=100.0):
    """Primary-vortex psi_min across grids; Richardson observed order."""
    rows = []
    for N in (33, 65, 129, 257):
        s = CavitySolver(N=N, Re=Re, mode="lid")
        s.run(tol=1e-7, max_iter=400000, check_every=500)
        rows.append((N, 1.0 / (N - 1), float(s.psi.min())))
        print(f"  conv Re={Re} N={N} psi_min={s.psi.min():.6f} "
              f"iters={s.iters}")
    rows = np.array(rows)
    # Richardson order using three finest grids (ratio-of-differences)
    f = rows[:, 2]
    # successive differences with constant refinement ratio r=2
    d = np.diff(f)
    p_richardson = np.log(np.abs(d[:-1] / d[1:])) / np.log(2.0)
    # Richardson-extrapolated value from finest triple (r=2, p~2)
    p_fine = p_richardson[-1]
    f_ext = f[-1] + (f[-1] - f[-2]) / (2.0**p_fine - 1.0)
    return rows, p_richardson, f_ext


if __name__ == "__main__":
    print("MMS Laplacian convergence:")
    mms, p_mms = mms_laplacian_order()
    np.savetxt(os.path.join(RES, "convergence_mms.csv"), mms,
               delimiter=",", header="N,h,L2_error", comments="")
    print("  N,h,L2err:\n", mms)
    print("  observed orders:", p_mms)

    print("Primary-vortex psi self-convergence (Re=100):")
    psi, p_rich, f_ext = psi_self_convergence(100.0)
    np.savetxt(os.path.join(RES, "convergence_psi.csv"), psi,
               delimiter=",", header="N,h,psi_min", comments="")

    summary = {
        "mms_orders": [float(x) for x in p_mms],
        "mms_order_mean": float(np.mean(p_mms[-3:])),
        "psi_richardson_orders": [float(x) for x in p_rich],
        "psi_observed_order": float(p_rich[-1]),
        "psi_extrapolated": float(f_ext),
        "psi_finest": float(psi[-1, 2]),
    }
    with open(os.path.join(RES, "convergence_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("Summary:", json.dumps(summary, indent=2))
