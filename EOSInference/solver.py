"""
solver.py -- your TOV + tidal (y-equation) solver, adapted for the hybrid EOS.

Same equations and dimensionless variables as neutronstar_solver.py
(Rbar = r/R0, Mbar = M/Msun, Pbar = P/P0, ebar = eps/P0, y(0)=2), with:
  * numba-compiled adaptive Dormand-Prince RK45 (no scipy in the inner loop),
  * the integration is split exactly at each transition pressure, and across
    each jump the tidal function gets the corrected Takatsy & Kovacs (2020)
    Eq. (11) condition   y+ - y- = - 4 pi r^3 d_eps /(m + 4 pi r^3 P)
    which in these variables is   dy = - Rbar^3 d_ebar / (Mbar + Rbar^3 Pbar),
  * stability of every star decided by the Bardeen-Thorne-Meltzer turning-point
    rule, so ALL stable branches (twin / third family) are kept,
  * the central-pressure sweep is adaptive (clusters around transitions and
    refinement around every extremum of M(Pc)).
Lambda = (2/3) k2 / beta^5 (identical to lambda / M_geom^5 in your solver).
"""
import numpy as np
from numba import njit
from scipy.interpolate import PchipInterpolator

import config
from hybrid_eos import P0, R0, BIG, HadronicTable, HybridEOS

# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------
@njit(cache=True)
def eos_eps_cs2(P, phase, par, hx, hy, hd):
    """eps-bar and cs^2 on the given phase (0 hadronic, 1 polytrope 1, 2 polytrope 2)."""
    if phase == 0:
        x = np.log(max(P, np.exp(hx[0])))
        n = hx.shape[0]
        k = np.searchsorted(hx, x) - 1
        if k < 0:
            k = 0
        if k > n - 2:
            k = n - 2
        h = hx[k + 1] - hx[k]
        t = (x - hx[k]) / h
        t2 = t * t
        t3 = t2 * t
        ly = ((2 * t3 - 3 * t2 + 1) * hy[k] + (t3 - 2 * t2 + t) * h * hd[k]
              + (-2 * t3 + 3 * t2) * hy[k + 1] + (t3 - t2) * h * hd[k + 1])
        dly = ((6 * t2 - 6 * t) / h * hy[k] + (3 * t2 - 4 * t + 1) * hd[k]
               + (-6 * t2 + 6 * t) / h * hy[k + 1] + (3 * t2 - 2 * t) * hd[k + 1])
        eps = np.exp(ly)
        if dly <= 0.0:
            return eps, 1e10
        return eps, (P / eps) / dly if P > 0 else (np.exp(x) / eps) / dly
    elif phase == 1:
        Pp = max(P, 1e-300)
        eps = (Pp / par[2]) ** (1.0 / par[3])
        return eps, par[3] * Pp / eps
    else:
        Pp = max(P, 1e-300)
        eps = (Pp / par[4]) ** (1.0 / par[5])
        return eps, par[5] * Pp / eps


@njit(cache=True)
def rhs(R, M, P, y, phase, par, hx, hy, hd):
    eps, cs2 = eos_eps_cs2(P, phase, par, hx, hy, hd)
    dM = eps * R * R
    dP = -(M * eps / (2 * R * R)) * (1 + P / eps) * (1 + R ** 3 * P / M) / (1 - M / R)
    elam = 1.0 / (1.0 - M / R)
    y_term = 1.0 + 0.5 * R * R * (P - eps)
    Qb = 5 * eps + 9 * P + (eps + P) / cs2
    r2Q = 0.5 * R * R * elam * Qb - 6 * elam - (elam * elam / (R * R)) * (M + R ** 3 * P) ** 2
    dy = -(1.0 / R) * (y * y + y * elam * y_term + r2Q)
    return dM, dP, dy


@njit(cache=True)
def dopri_step(R, M, P, y, h, phase, par, hx, hy, hd):
    """One Dormand-Prince 5(4) step; returns new state and the 4th/5th order difference."""
    k1M, k1P, k1y = rhs(R, M, P, y, phase, par, hx, hy, hd)
    k2M, k2P, k2y = rhs(R + h / 5, M + h * k1M / 5, P + h * k1P / 5, y + h * k1y / 5, phase, par, hx, hy, hd)
    k3M, k3P, k3y = rhs(R + 3 * h / 10, M + h * (3 * k1M + 9 * k2M) / 40, P + h * (3 * k1P + 9 * k2P) / 40,
                        y + h * (3 * k1y + 9 * k2y) / 40, phase, par, hx, hy, hd)
    k4M, k4P, k4y = rhs(R + 4 * h / 5, M + h * (44 * k1M - 168 * k2M + 160 * k3M) / 45,
                        P + h * (44 * k1P - 168 * k2P + 160 * k3P) / 45,
                        y + h * (44 * k1y - 168 * k2y + 160 * k3y) / 45, phase, par, hx, hy, hd)
    k5M, k5P, k5y = rhs(R + 8 * h / 9,
                        M + h * (19372 * k1M - 76080 * k2M + 64448 * k3M - 1908 * k4M) / 6561,
                        P + h * (19372 * k1P - 76080 * k2P + 64448 * k3P - 1908 * k4P) / 6561,
                        y + h * (19372 * k1y - 76080 * k2y + 64448 * k3y - 1908 * k4y) / 6561,
                        phase, par, hx, hy, hd)
    k6M, k6P, k6y = rhs(R + h,
                        M + h * (9017 * k1M / 3168 - 355 * k2M / 33 + 46732 * k3M / 5247 + 49 * k4M / 176
                                 - 5103 * k5M / 18656),
                        P + h * (9017 * k1P / 3168 - 355 * k2P / 33 + 46732 * k3P / 5247 + 49 * k4P / 176
                                 - 5103 * k5P / 18656),
                        y + h * (9017 * k1y / 3168 - 355 * k2y / 33 + 46732 * k3y / 5247 + 49 * k4y / 176
                                 - 5103 * k5y / 18656),
                        phase, par, hx, hy, hd)
    a1, a3, a4, a5, a6 = 35.0 / 384, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84
    Mn = M + h * (a1 * k1M + a3 * k3M + a4 * k4M + a5 * k5M + a6 * k6M)
    Pn = P + h * (a1 * k1P + a3 * k3P + a4 * k4P + a5 * k5P + a6 * k6P)
    yn = y + h * (a1 * k1y + a3 * k3y + a4 * k4y + a5 * k5y + a6 * k6y)
    k7M, k7P, k7y = rhs(R + h, Mn, Pn, yn, phase, par, hx, hy, hd)
    b1, b3, b4, b5, b6, b7 = 5179.0 / 57600, 7571.0 / 16695, 393.0 / 640, -92097.0 / 339200, 187.0 / 2100, 1.0 / 40
    eM = h * ((a1 - b1) * k1M + (a3 - b3) * k3M + (a4 - b4) * k4M + (a5 - b5) * k5M + (a6 - b6) * k6M - b7 * k7M)
    eP = h * ((a1 - b1) * k1P + (a3 - b3) * k3P + (a4 - b4) * k4P + (a5 - b5) * k5P + (a6 - b6) * k6P - b7 * k7P)
    ey = h * ((a1 - b1) * k1y + (a3 - b3) * k3y + (a4 - b4) * k4y + (a5 - b5) * k5y + (a6 - b6) * k6y - b7 * k7y)
    return Mn, Pn, yn, eM, eP, ey


@njit(cache=True)
def k2_love(beta, y):
    """k2(beta, y_R), Eq. (2) of Postnikov et al.; plain double precision is accurate for
    beta >~ 5e-3 (validated against your mpmath version); returns nan below that."""
    if beta < 0.004:
        return np.nan
    b = beta
    t1 = 2.0 - y + 2.0 * b * (y - 1.0)
    br = (2 * b * (6 - 3 * y + 3 * b * (5 * y - 8)
                   + 2 * b ** 2 * (13 - 11 * y + b * (3 * y - 2) + 2 * b ** 2 * (1 + y)))
          + 3 * (1 - 2 * b) ** 2 * t1 * np.log1p(-2 * b))
    return 8.0 / 5.0 * b ** 5 * (1 - 2 * b) ** 2 * t1 / br


@njit(cache=True)
def integrate_star(Pc, par, hx, hy, hd, Psurf, rtol, atol_M, atol_P, atol_y):
    """Integrate one star. Returns (status, Rbar, Mbar, yR, eps_c).  status 0 = ok."""
    P_t, P_1 = par[0], par[1]
    if Pc > P_1:
        phase = 2
    elif Pc > P_t:
        phase = 1
    else:
        phase = 0
    eps_c, _ = eos_eps_cs2(Pc, phase, par, hx, hy, hd)
    if phase == 2:
        thr = P_1
    elif phase == 1:
        thr = P_t
    else:
        thr = Psurf
    R = 1e-6
    M = eps_c * R ** 3 / 3.0
    P = Pc
    y = 2.0
    h = 1e-6
    for it in range(400000):
        Mn, Pn, yn, eM, eP, ey = dopri_step(R, M, P, y, h, phase, par, hx, hy, hd)
        sc_M = atol_M + rtol * max(abs(M), abs(Mn))
        sc_P = atol_P + rtol * max(abs(P), abs(Pn))
        sc_y = atol_y + rtol * max(abs(y), abs(yn))
        err = max(abs(eM) / sc_M, abs(eP) / sc_P, abs(ey) / sc_y)
        if not np.isfinite(err):
            return 1, 0.0, 0.0, 0.0, eps_c
        if err > 1.0:
            h *= max(0.2, 0.9 * err ** (-0.2))
            if h < 1e-14:
                return 2, 0.0, 0.0, 0.0, eps_c
            continue
        if Pn <= thr:
            # locate the crossing P = thr by bisection on the step length
            lo = 0.0
            hi = h
            for k in range(60):
                mid = 0.5 * (lo + hi)
                Mm, Pm, ym, a, b, c = dopri_step(R, M, P, y, mid, phase, par, hx, hy, hd)
                if Pm > thr:
                    lo = mid
                else:
                    hi = mid
            hh = 0.5 * (lo + hi)
            Mm, Pm, ym, a, b, c = dopri_step(R, M, P, y, hh, phase, par, hx, hy, hd)
            R = R + hh
            M = Mm
            y = ym
            P = thr
            if phase == 0:
                return 0, R, M, y, eps_c
            # first-order transition: eps drops outward by d across r_d;
            # corrected Takatsy-Kovacs jump condition (Eq. 11)
            d = par[7] if phase == 2 else par[6]
            # par[8] = 1 switches to the ORIGINAL Postnikov et al. form (no P term); validation only
            y = y - R ** 3 * d / (M + R ** 3 * P * (1.0 - par[8]))
            phase -= 1
            thr = P_t if phase == 1 else Psurf
            h = max(hh, 1e-8)
            continue
        R += h
        M, P, y = Mn, Pn, yn
        h *= min(5.0, max(0.2, 0.9 * err ** (-0.2))) if err > 0 else 5.0
    return 3, 0.0, 0.0, 0.0, eps_c


@njit(cache=True)
def sweep_kernel(Pcs, par, hx, hy, hd, Psurf, rtol, atol_M, atol_P, atol_y):
    n = Pcs.shape[0]
    out = np.full((n, 7), np.nan)  # Pc, eps_c(bar), R[km], M, beta, yR, k2
    for i in range(n):
        st, Rb, Mb, yR, ec = integrate_star(Pcs[i], par, hx, hy, hd, Psurf, rtol, atol_M, atol_P, atol_y)
        if st != 0:
            continue
        beta = Mb / (2.0 * Rb)
        k2 = k2_love(beta, yR)
        out[i, 0] = Pcs[i]
        out[i, 1] = ec
        out[i, 2] = Rb * 2.9533393820668783
        out[i, 3] = Mb
        out[i, 4] = beta
        out[i, 5] = yR
        out[i, 6] = k2
    return out


# ---------------------------------------------------------------------------
# python-side: sweep, stability, branches
# ---------------------------------------------------------------------------
def _sweep(eos, Pcs, cfg):
    t = eos.table
    return sweep_kernel(np.ascontiguousarray(Pcs), eos.par, t.x, t.y, t.d, t.Pbar_min,
                        cfg["rtol"], cfg["atol_M"], cfg["atol_P"], cfg["atol_y"])


def _mass_noise(M, cfg):
    """Absolute mass difference below which dM carries no information.

    The adaptive RK45 controls the local error per step at `rtol`; the accumulated
    error in M is empirically a few times rtol*M (see validate.py test 4).  We take
    a deliberately generous multiple so that no real branch -- which must span
    `min_branch_dM` in total -- is ever confused with noise.
    """
    return cfg["noise_factor"] * cfg["rtol"] * max(float(np.max(M)), 1e-3)


def _dM_sign(M, tol):
    """Sign of dM/dPc with a dead-band.

    Segments whose |dM| is at or below the integrator's noise floor carry no
    information about stability; they inherit the sign of the nearest resolved
    segment to their left (or to their right if the run starts flat).  Without
    this, adaptive refinement eventually brackets every extremum so tightly
    that consecutive masses differ by less than the ODE tolerance and the raw
    sign of dM flips at random -- which used to manufacture spurious extrema
    and destroy real branches.
    """
    dM = np.diff(M)
    s = np.sign(dM)
    s[np.abs(dM) <= tol] = 0
    nz = np.flatnonzero(s)
    if nz.size == 0:
        return s, dM
    # forward-fill, then back-fill the leading flat run
    idx = np.maximum.accumulate(np.where(s != 0, np.arange(s.size), -1))
    lead = idx < 0
    out = np.where(lead, s[nz[0]], s[np.maximum(idx, 0)])
    return out, dM


def _extrema_insert(Pc, M, tol, min_dlnP):
    """Geometric midpoints around every interior extremum of M(Pc).

    Refinement stops once a bracket is narrower than `min_dlnP` in ln Pc, or
    once the mass difference across it is at the noise floor: past that point
    more points only add noise (see _dM_sign).
    """
    s, dM = _dM_sign(M, tol)
    new = []
    for i in range(1, len(M) - 1):
        if s[i - 1] * s[i] < 0:
            for j in (i - 1, i):
                lo, hi = Pc[j], Pc[j + 1]
                if np.log(hi / lo) < min_dlnP or abs(dM[j]) <= tol:
                    continue
                new.extend(np.exp(np.linspace(np.log(lo), np.log(hi), 5)[1:-1]))
    return np.array(new)


def _thin_grid(Pc, arr, min_dlnP):
    """Drop points that sit closer than min_dlnP in ln Pc to the one kept before them."""
    keep = np.ones(len(Pc), bool)
    last = np.log(Pc[0])
    for i in range(1, len(Pc)):
        lp = np.log(Pc[i])
        if lp - last < min_dlnP:
            keep[i] = False
        else:
            last = lp
    return Pc[keep], arr[keep]


def build_Pc_grid(eos, cfg):
    t = eos.table
    P_lo = float(t.P_bar_of_eps_bar(cfg["eps_c_min"] / P0))
    P_hi = eos.Pbar_at_eps_central(cfg["eps_c_max"])
    grid = list(np.geomspace(P_lo, P_hi, cfg["n_base"]))
    for T in (eos.P_t, eos.P_1):
        if P_lo < T < P_hi:
            for dlt in cfg["cluster_deltas"]:
                if dlt >= cfg["min_dlnPc"]:      # finer than this is pure ODE noise
                    grid += [T * (1 + dlt), T * (1 - dlt)]
    return np.unique(np.array(grid)), P_lo, P_hi


class Sequence:
    """A computed stellar sequence with its stable branches."""

    def __init__(self):
        self.ok = False
        self.reason = ""

    # values on every stable branch containing mass m (rows = branches, NaN if absent)
    def branch_values(self, m, key):
        m = np.atleast_1d(m)
        out = np.full((len(self.branches), m.size), np.nan)
        for b, br in enumerate(self.branches):
            lo, hi = br["M"][0], br["M"][-1]
            sel = (m >= lo) & (m <= hi)
            if np.any(sel):
                out[b, sel] = br["interp"][key](m[sel])
                if key == "Lam":
                    out[b, sel] = np.exp(out[b, sel])
        return out


def analyze_sequence(eos, arr, cfg):
    """Turn raw sweep rows into branches using the turning-point stability rule."""
    seq = Sequence()
    good = np.isfinite(arr[:, 3]) & np.isfinite(arr[:, 6]) & (arr[:, 6] > 0)
    arr = arr[good]
    seq.n_points = len(arr)
    if len(arr) < 8:
        seq.reason = "too few valid stars"
        return seq
    Pc, ec, R, M, beta, yR, k2 = arr.T
    Lam = (2.0 / 3.0) * k2 / beta ** 5
    # Stability.  Along a one-parameter sequence ordered by central pressure the
    # Bardeen-Thorne-Meltzer turning-point theorem says exactly one radial mode
    # changes stability at each extremum of M(Pc).  Extrema of a continuous curve
    # alternate max/min, and the sequence starts stable, so the number of unstable
    # modes is 0 on every rising segment and 1 on every falling one: the counting
    # reduces to dM/dPc > 0.  That is what we use -- it is the same criterion as
    # the mode count but has no state to accumulate an error in, which matters
    # because the mass differences near an extremum reach the ODE noise floor.
    tol = _mass_noise(M, cfg)
    sgn, dM = _dM_sign(M, tol)
    if sgn[0] <= 0:
        seq.reason = "lowest-density star already unstable (lower eps_c_min)"
        return seq
    extrema = [(i, "max" if sgn[i - 1] > 0 else "min") for i in range(1, len(sgn))
               if sgn[i - 1] * sgn[i] < 0]
    stable_seg = sgn > 0
    # runs of stable segments -> branches
    branches, i = [], 0
    while i < len(dM):
        if stable_seg[i]:
            j = i
            while j + 1 < len(dM) and stable_seg[j + 1]:
                j += 1
            idx = np.arange(i, j + 2)
            # the dead-band in _dM_sign lets a branch contain sub-segments whose
            # mass is flat to within the ODE noise; keep a strictly increasing
            # subset so M is a valid interpolation abscissa
            keep = [idx[0]]
            for q in idx[1:]:
                if M[q] > M[keep[-1]]:
                    keep.append(q)
            idx = np.array(keep)
            if len(idx) >= 4 and M[idx[-1]] - M[idx[0]] >= cfg["min_branch_dM"]:
                br = dict(idx=idx, Pc=Pc[idx], eps_c=ec[idx] * P0, R=R[idx], M=M[idx], Lam=Lam[idx], k2=k2[idx])
                br["interp"] = {"R": PchipInterpolator(br["M"], br["R"]),
                                "Lam": PchipInterpolator(br["M"], np.log(br["Lam"]))}
                branches.append(br)
            i = j + 1
        else:
            i += 1
    if not branches:
        seq.reason = "no stable branch"
        return seq
    seq.Pc, seq.eps_c, seq.R, seq.M, seq.Lam, seq.k2 = Pc, ec * P0, R, M, Lam, k2
    seq.branches = branches
    seq.n_branches = len(branches)
    seq.extrema = extrema
    # M_TOV: highest mass on any stable branch, refined by a local parabola in Pc
    top = max(branches, key=lambda b: b["M"][-1])
    j = top["idx"][-1]
    Mtov, Pc_tov = M[j], Pc[j]
    if 1 <= j < len(M) - 1 and np.log(Pc[j + 1] / Pc[j - 1]) > 5e-3:   # skip when neighbours are a tight cluster
        lo, hi = j - 1, j + 2
        x = np.log(Pc[lo:hi])
        x0 = x[1]                                   # centre the abscissa: avoids ill-conditioned fits
        a, b, c = np.polyfit(x - x0, M[lo:hi], 2)
        if a < 0:
            xs = -b / (2 * a)
            if x[0] - x0 <= xs <= x[-1] - x0:
                Mtov, Pc_tov = a * xs ** 2 + b * xs + c, np.exp(xs + x0)
    seq.M_tov = float(max(Mtov, M[j]))
    seq.Pc_tov = float(Pc_tov)
    seq.eps_c_tov = float(top["eps_c"][-1])
    seq.Pc_max_stable = max(b["Pc"][-1] for b in branches)
    seq.cs2_max = eos.max_cs2_up_to(seq.Pc_max_stable)
    seq.ok = True
    return seq


def compute_sequence(eos, cfg=None):
    """Full pipeline: adaptive Pc grid -> sweep -> refine extrema -> branches."""
    cfg = cfg or config.SWEEP
    if not eos.valid:
        s = Sequence()
        s.reason = eos.reason
        return s
    Pcs, P_lo, P_hi = build_Pc_grid(eos, cfg)
    arr = _sweep(eos, Pcs, cfg)
    for _ in range(cfg["max_extensions"]):          # extend upward while M still rises at the top
        good = np.isfinite(arr[:, 3])
        if good.sum() > 2 and arr[good][-1, 3] > arr[good][-2, 3] and Pcs[-1] < 0.9 * eos.table.Pbar_max * 1e3:
            ext = np.geomspace(Pcs[-1], Pcs[-1] * 2.5, 12)[1:]
            Pcs = np.concatenate([Pcs, ext])
            arr = np.vstack([arr, _sweep(eos, ext, cfg)])
        else:
            break
    for _ in range(cfg["refine_rounds"]):
        good = np.isfinite(arr[:, 3])
        new = _extrema_insert(Pcs[good], arr[good, 3],
                              _mass_noise(arr[good, 3], cfg), cfg["min_dlnPc"])
        if new.size == 0:
            break
        Pcs = np.concatenate([Pcs, new])
        arr = np.vstack([arr, _sweep(eos, new, cfg)])
        order = np.argsort(Pcs)
        Pcs, arr = Pcs[order], arr[order]
    # never hand analyze_sequence two stars closer than the resolution at which
    # their mass difference is still meaningful
    Pcs, arr = _thin_grid(Pcs, arr, cfg["min_dlnPc"])
    return analyze_sequence(eos, arr, cfg)
