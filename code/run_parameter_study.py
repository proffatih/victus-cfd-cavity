"""
Parameter studies.

(A) Lid-driven cavity Reynolds-number sweep: primary-vortex strength and
    location, and the bottom-right / bottom-left secondary corner vortices,
    as functions of Re. Outputs results/re_sweep.csv.

(B) Natural-convection (Boussinesq) Rayleigh-number sweep at Pr=0.71:
    average hot-wall Nusselt number Nu vs Ra, compared to the de Vahl Davis
    (1983) benchmark, and a fitted power-law correlation Nu = C Ra^n.
    Outputs results/ra_sweep.csv and results/natural_fields.npz.
"""
import os
import json
import numpy as np

from solver import CavitySolver

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
os.makedirs(RES, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

# de Vahl Davis (1983) benchmark average Nusselt numbers (hot wall), Pr=0.71
DVD_NU = {1e3: 1.118, 1e4: 2.243, 1e5: 4.519, 1e6: 8.800}


def corner_vortex_strength(psi, x, y, region):
    """Max positive (clockwise-opposing) psi in a corner sub-region."""
    if region == "BR":  # bottom-right
        ix = x > 0.5
        iy = y < 0.5
    elif region == "BL":  # bottom-left
        ix = x < 0.5
        iy = y < 0.5
    sub = psi[np.ix_(ix, iy)]
    return float(sub.max())


def re_sweep():
    rows = []
    # (Re, N, tol, max_iter) -- bounded per case so the sweep always finishes
    cases = [
        (100, 129, 1e-6, 120000),
        (400, 129, 1e-6, 120000),
        (1000, 129, 1e-6, 150000),
        (2000, 129, 1e-6, 150000),
        (3200, 129, 1e-6, 180000),
        (5000, 161, 1e-6, 200000),
        (7500, 161, 1e-6, 200000),
    ]
    csv_path = os.path.join(RES, "re_sweep.csv")
    with open(csv_path, "w") as fh:
        fh.write("Re,psi_primary,xc,yc,psi_BR,psi_BL,iters,N\n")
    for Re, N, tol, mi in cases:
        s = CavitySolver(N=N, Re=float(Re), mode="lid")
        s.run(tol=tol, max_iter=mi, check_every=500)
        idx = np.unravel_index(np.argmin(s.psi), s.psi.shape)
        psi_p = float(s.psi[idx])
        xc, yc = float(s.x[idx[0]]), float(s.y[idx[1]])
        br = corner_vortex_strength(s.psi, s.x, s.y, "BR")
        bl = corner_vortex_strength(s.psi, s.x, s.y, "BL")
        rows.append((Re, psi_p, xc, yc, br, bl, s.iters, N))
        with open(csv_path, "a") as fh:
            fh.write(f"{Re},{psi_p},{xc},{yc},{br},{bl},{s.iters},{N}\n")
        print(f"  Re={Re:5d} N={N} psi_p={psi_p:.5f} "
              f"centre=({xc:.3f},{yc:.3f}) BR={br:.3e} BL={bl:.3e} "
              f"iters={s.iters}", flush=True)
    return np.array(rows)


def ra_sweep():
    rows = []
    fields = {}
    # (Ra, N, tol, max_iter)
    cases = [
        (1e3, 81, 1e-6, 200000),
        (1e4, 101, 1e-6, 250000),
        (1e5, 129, 1e-6, 300000),
        (1e6, 161, 1e-6, 400000),
    ]
    csv_path = os.path.join(RES, "ra_sweep.csv")
    with open(csv_path, "w") as fh:
        fh.write("Ra,Nu_solver,Nu_ref,err_pct,psi_max,iters\n")
    for Ra, N, tol, mi in cases:
        s = CavitySolver(N=N, Ra=Ra, Pr=0.71, mode="natural")
        s.run(tol=tol, max_iter=mi, check_every=500)
        nu = s.nusselt_left()
        ref = DVD_NU[Ra]
        err = 100.0 * abs(nu - ref) / ref
        psi_max = float(np.abs(s.psi).max())
        rows.append((Ra, nu, ref, err, psi_max, s.iters))
        fields[f"Ra{int(Ra)}"] = dict(
            x=s.x, y=s.y, psi=s.psi.copy(), theta=s.theta.copy())
        with open(csv_path, "a") as fh:
            fh.write(f"{Ra},{nu},{ref},{err},{psi_max},{s.iters}\n")
        print(f"  Ra={Ra:.0e} N={N} Nu={nu:.3f} (ref {ref:.3f}, {err:.2f}%) "
              f"|psi|max={psi_max:.3f}", flush=True)
    rows = np.array(rows)
    # power-law fit Nu = C Ra^n
    logRa = np.log10(rows[:, 0])
    logNu = np.log10(rows[:, 1])
    n, logC = np.polyfit(logRa, logNu, 1)
    C = 10.0**logC
    # save fields
    np.savez_compressed(
        os.path.join(DATA, "natural_fields.npz"),
        **{f"{k}_{q}": v for k, d in fields.items() for q, v in d.items()})
    return rows, C, n


if __name__ == "__main__":
    print("(A) Reynolds sweep (lid-driven cavity):")
    re_rows = re_sweep()
    print("(B) Rayleigh sweep (natural convection):")
    ra_rows, C, n = ra_sweep()
    summary = {
        "re_list": [int(r) for r in re_rows[:, 0]],
        "nu_correlation": {"C": float(C), "n": float(n),
                           "form": "Nu = C * Ra^n"},
        "ra_results": [
            {"Ra": float(r[0]), "Nu": float(r[1]), "Nu_ref": float(r[2]),
             "err_pct": float(r[3])} for r in ra_rows],
    }
    with open(os.path.join(RES, "parameter_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Nu correlation: Nu = {C:.3f} * Ra^{n:.3f}")
    print("Saved parameter_summary.json")
