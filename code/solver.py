"""
Vorticity-streamfunction finite-difference solver for 2-D incompressible
Navier-Stokes flow in a square cavity, with an optional Boussinesq thermal
coupling for natural (buoyancy-driven) convection.

Author: Fatih Gul (Recep Tayyip Erdogan University)
Paper:  Victus_Pardus_0008_CFD_Cavity / Computational Mechanics

Formulation (non-dimensional, square cavity [0,1]x[0,1]):

  Vorticity transport:
     d(omega)/dt + u d(omega)/dx + v d(omega)/dy = (1/Re) lap(omega) + buoyancy
  Streamfunction (Poisson):
     lap(psi) = -omega,   u = d(psi)/dy,  v = -d(psi)/dx

For natural convection the velocity scale is the thermal diffusion velocity,
the momentum equation uses the Prandtl number Pr and the buoyancy source is
  Ra*Pr * d(theta)/dx
and the energy equation
     d(theta)/dt + u d(theta)/dx + v d(theta)/dy = lap(theta)
is advanced alongside.

Time integration: explicit Euler for the vorticity/energy transport (with a
CFL/diffusion-stable step), Thom's first-order wall vorticity, and a fast
direct Poisson solve for the streamfunction via the 2-D discrete sine
transform (DST), which is exact for the 5-point Laplacian with homogeneous
Dirichlet BC on psi.

No external CFD library is used; only numpy/scipy.
"""

import numpy as np
from scipy.fft import dstn, idstn


def _poisson_dst(rhs, h):
    """Solve lap(psi) = rhs on interior nodes with psi=0 on all walls.

    Uses the 2-D type-I discrete sine transform, which diagonalises the
    standard 5-point Laplacian with homogeneous Dirichlet boundaries.
    `rhs` is the (Nint x Nint) interior right-hand side; returns interior psi.
    """
    n = rhs.shape[0]
    # eigenvalues of 1-D second difference with Dirichlet BC
    k = np.arange(1, n + 1)
    lam = (2.0 * np.cos(np.pi * k / (n + 1)) - 2.0) / h**2
    denom = lam[:, None] + lam[None, :]
    rhs_hat = dstn(rhs, type=1)
    psi_hat = rhs_hat / denom
    psi = idstn(psi_hat, type=1)
    return psi


class CavitySolver:
    """Lid-driven (isothermal) or natural-convection square-cavity solver."""

    def __init__(self, N, Re=100.0, Ra=None, Pr=0.71, mode="lid"):
        """
        N    : number of grid points per side (including boundaries)
        Re   : Reynolds number (lid-driven mode)
        Ra   : Rayleigh number (natural-convection mode)
        Pr   : Prandtl number (natural-convection mode)
        mode : 'lid' or 'natural'
        """
        self.N = N
        self.mode = mode
        self.Re = float(Re)
        self.Ra = None if Ra is None else float(Ra)
        self.Pr = float(Pr)
        self.h = 1.0 / (N - 1)
        x = np.linspace(0.0, 1.0, N)
        self.x = x
        self.y = x.copy()
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")  # X[i,j]=x_i

        self.psi = np.zeros((N, N))
        self.omega = np.zeros((N, N))
        # temperature: hot left wall theta=1, cold right wall theta=0 (natural)
        self.theta = np.zeros((N, N))
        if mode == "natural":
            self.theta[0, :] = 1.0
            self.theta[-1, :] = 0.0
            # linear initial guess
            self.theta[:, :] = 1.0 - self.X

    # ---- velocity from streamfunction (central differences) ----
    def velocities(self):
        h = self.h
        u = np.zeros_like(self.psi)
        v = np.zeros_like(self.psi)
        psi = self.psi
        u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)   # d psi/dy
        v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)  # -d psi/dx
        # lid BC for lid-driven cavity: top wall u=1
        if self.mode == "lid":
            u[:, -1] = 1.0
        return u, v

    def _apply_wall_vorticity(self):
        """Thom's formula for wall vorticity (first order)."""
        psi, h = self.psi, self.h
        w = self.omega
        # bottom (j=0), top (j=N-1), left (i=0), right (i=N-1)
        w[:, 0] = -2.0 * psi[:, 1] / h**2
        w[:, -1] = -2.0 * psi[:, -2] / h**2
        w[0, :] = -2.0 * psi[1, :] / h**2
        w[-1, :] = -2.0 * psi[-2, :] / h**2
        if self.mode == "lid":
            # moving top lid (u=1): omega_wall = -2 psi/h^2 - 2 U/h
            w[:, -1] = -2.0 * psi[:, -2] / h**2 - 2.0 * 1.0 / h

    def step(self, dt):
        h = self.h
        u, v = self.velocities()
        w = self.omega

        # Laplacian of omega (interior)
        lap_w = (
            w[2:, 1:-1] + w[:-2, 1:-1] + w[1:-1, 2:] + w[1:-1, :-2]
            - 4.0 * w[1:-1, 1:-1]
        ) / h**2
        # advection (central differences)
        dwdx = (w[2:, 1:-1] - w[:-2, 1:-1]) / (2 * h)
        dwdy = (w[1:-1, 2:] - w[1:-1, :-2]) / (2 * h)
        ui = u[1:-1, 1:-1]
        vi = v[1:-1, 1:-1]

        if self.mode == "lid":
            nu = 1.0 / self.Re
            adv = ui * dwdx + vi * dwdy
            w_new = w[1:-1, 1:-1] + dt * (-adv + nu * lap_w)
        else:
            th = self.theta
            # momentum: nu term = Pr * lap(omega), buoyancy = Ra*Pr * d(theta)/dx
            dthdx = (th[2:, 1:-1] - th[:-2, 1:-1]) / (2 * h)
            adv = ui * dwdx + vi * dwdy
            w_new = w[1:-1, 1:-1] + dt * (
                -adv + self.Pr * lap_w + self.Ra * self.Pr * dthdx
            )
        self.omega[1:-1, 1:-1] = w_new

        # ---- energy equation (natural convection) ----
        if self.mode == "natural":
            th = self.theta
            lap_t = (
                th[2:, 1:-1] + th[:-2, 1:-1] + th[1:-1, 2:] + th[1:-1, :-2]
                - 4.0 * th[1:-1, 1:-1]
            ) / h**2
            dtdx = (th[2:, 1:-1] - th[:-2, 1:-1]) / (2 * h)
            dtdy = (th[1:-1, 2:] - th[1:-1, :-2]) / (2 * h)
            adv_t = ui * dtdx + vi * dtdy
            self.theta[1:-1, 1:-1] = th[1:-1, 1:-1] + dt * (-adv_t + lap_t)
            # BC: hot/cold vertical walls (Dirichlet), adiabatic top/bottom
            self.theta[0, :] = 1.0
            self.theta[-1, :] = 0.0
            self.theta[:, 0] = self.theta[:, 1]    # dtheta/dy = 0
            self.theta[:, -1] = self.theta[:, -2]

        # ---- Poisson solve for streamfunction ----
        rhs = -self.omega[1:-1, 1:-1]
        self.psi[1:-1, 1:-1] = _poisson_dst(rhs, h)
        # psi=0 on walls already enforced (array stays zero on boundary)

        # ---- update wall vorticity ----
        self._apply_wall_vorticity()

    def stable_dt(self, safety=0.2):
        h = self.h
        if self.mode == "lid":
            nu = 1.0 / self.Re
            dt_diff = 0.25 * h**2 / nu
            dt_conv = h / 1.0
        else:
            nu = self.Pr
            dt_diff = 0.25 * h**2 / max(nu, 1.0)
            # convective velocity scale ~ sqrt(Ra)*Pr is conservative
            ucfl = max(np.sqrt(self.Ra) * self.Pr, 1.0)
            dt_conv = h / ucfl
        return safety * min(dt_diff, dt_conv)

    def run(self, tol=1e-6, max_iter=200000, dt=None, check_every=200,
            verbose=False):
        if dt is None:
            dt = self.stable_dt()
        prev = self.omega.copy()
        history = []
        for it in range(1, max_iter + 1):
            self.step(dt)
            if it % check_every == 0:
                diff = np.linalg.norm(self.omega - prev) / (
                    np.linalg.norm(self.omega) + 1e-30
                )
                rate = diff / (dt * check_every)
                history.append((it, rate))
                if verbose:
                    print(f"  it={it:6d}  d(omega)/dt~{rate:.3e}")
                if rate < tol:
                    break
                prev = self.omega.copy()
        self.iters = it
        self.dt = dt
        self.res_history = history
        return it

    # ---- derived quantities ----
    def nusselt_left(self):
        """Average Nusselt number on the hot (left) wall."""
        h = self.h
        th = self.theta
        # one-sided second-order derivative -dtheta/dx at i=0
        dtdx = (-3 * th[0, :] + 4 * th[1, :] - th[2, :]) / (2 * h)
        nu_local = -dtdx  # heat flux into domain
        # integrate over y with trapezoid
        return np.trapezoid(nu_local, self.y)

    def streamfunction_extremum(self):
        return float(self.psi.min()), float(self.psi.max())
