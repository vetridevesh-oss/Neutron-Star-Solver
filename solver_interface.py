"""
solver_interface.py -- Fast, in-memory, nested-sampling-safe wrapper around
the TOV + tidal Love-number physics in neutronstar_solver.py.

WHY THIS EXISTS (see the mentee's earlier review conversation for the full
reasoning): EOStoObservables()/EOStoObservables_parallel() are built for
one-time, publication-quality M-R/Lambda-M curves -- a fixed 3000-point
central-pressure sweep per call, reading the EOS from a FILE. Nested
sampling needs O(1e4-5e4) likelihood calls, each on a DIFFERENT (in-memory)
EOS instance, and each likelihood call only needs a handful of numbers
(R, Lambda, k2 at specific target masses; Mmax). This module reuses the
*exact same* TOV/tidal ODE right-hand side and the *exact same*
love_number_k2 formula from neutronstar_solver.py (imported, not
reimplemented, so there's only one place the physics can go wrong), but:

  1. accepts (P, eps) arrays directly -- no file I/O per call
  2. finds target masses via 1-D root-finding (brentq) on the stable branch,
     not a full 3000-point sweep
  3. never raises on a bad draw -- returns None / a rejection flag so the
     nested sampler's likelihood function can turn it into -inf instead of
     crashing the whole run
  4. clips Pbar_eval on BOTH bounds (the original only clipped the lower
     bound; an adaptive RK45 step evaluating the RHS just above Pbar_max
     during a stiff crossover region would silently return NaN otherwise)
  5. exposes causality (cs^2 <= 1) as a queryable flag, not just an
     integration side-effect
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from neutronstar_solver import R0, P0, love_number_k2


# ---------------------------------------------------------------------------
# 1. Build an in-memory EOS bundle (replaces file-based _init_worker)
# ---------------------------------------------------------------------------
def build_bundle(P_arr, eps_arr):
    """P_arr, eps_arr in MeV/fm^3 (same convention as the tabulated files
    neutronstar_solver.py reads). Returns None if the table can't support a
    valid monotonic log-log interpolant (caller should treat as a rejected
    draw)."""
    P_arr = np.asarray(P_arr, dtype=float)
    eps_arr = np.asarray(eps_arr, dtype=float)
    order = np.argsort(P_arr)
    P_arr, eps_arr = P_arr[order], eps_arr[order]

    if not (np.all(np.diff(P_arr) > 0) and np.all(np.diff(eps_arr) > 0)):
        return None
    if np.any(P_arr <= 0) or np.any(eps_arr <= 0):
        return None

    Pbar = P_arr / P0
    ebar = eps_arr / P0
    logPbar, logebar = np.log(Pbar), np.log(ebar)
    logE_of_logP = PchipInterpolator(logPbar, logebar, extrapolate=False)

    return {
        "logE_of_logP": logE_of_logP,
        "dlogE_dlogP": logE_of_logP.derivative(1),
        "Pbar_min": Pbar.min(),
        "Pbar_max": Pbar.max(),
    }


def check_causality(bundle, n_check=500):
    """Vectorized cs^2 <= 1 and dP/deps >= 0 check across the table.
    Returns (causal_ok: bool, max_cs2: float)."""
    logP = np.linspace(np.log(bundle["Pbar_min"]), np.log(bundle["Pbar_max"]), n_check)
    slope = bundle["dlogE_dlogP"](logP)
    Pbar = np.exp(logP)
    ebar = np.exp(bundle["logE_of_logP"](logP))
    with np.errstate(divide="ignore", invalid="ignore"):
        cs2 = (Pbar / ebar) / slope
    finite = np.isfinite(cs2)
    if not np.any(finite):
        return False, np.nan
    max_cs2 = np.nanmax(cs2[finite])
    causal_ok = bool(np.all(slope[finite] > 0) and max_cs2 <= 1.0 + 1e-6)
    return causal_ok, float(max_cs2)


# ---------------------------------------------------------------------------
# 2. Single-star solve (same physics as _TOV_w/_collapsed_star_task, but
#    reading from a passed-in bundle rather than a process-global dict, and
#    with symmetric pressure clipping)
# ---------------------------------------------------------------------------
def _tov_rhs(Rbar, y, bundle):
    Mbar, Pbar, yL = y
    Pbar_eval = np.clip(Pbar, bundle["Pbar_min"], bundle["Pbar_max"])
    lp = np.log(Pbar_eval)
    ebar = np.exp(bundle["logE_of_logP"](lp))
    slope = bundle["dlogE_dlogP"](lp)
    cs2 = 1e10 if slope <= 0 else float((Pbar_eval / ebar) / slope)

    dMbar_dRbar = ebar * Rbar ** 2
    dPbar_dRbar = -1 * (Mbar * ebar / (2 * Rbar ** 2)) * (1 + Pbar / ebar) \
        * (1 + Rbar ** 3 * Pbar / Mbar) * (1 - Mbar / Rbar) ** -1

    ELAM = 1.0 / (1.0 - Mbar / Rbar)
    y_elam_term = 1.0 + 0.5 * Rbar ** 2 * (Pbar - ebar)
    Q_bracket = 5 * ebar + 9 * Pbar + (ebar + Pbar) / cs2
    r2Q = 0.5 * Rbar ** 2 * ELAM * Q_bracket - 6 * ELAM \
        - (ELAM ** 2 / Rbar ** 2) * (Mbar + Rbar ** 3 * Pbar) ** 2
    dyL_dRbar = -(1.0 / Rbar) * (yL ** 2 + yL * ELAM * y_elam_term + r2Q)

    return [dMbar_dRbar, dPbar_dRbar, dyL_dRbar]


def _surface(Rbar, y, bundle):
    return y[1]


def solve_star(Pbar_c, bundle):
    """One TOV+tidal solve. Returns dict(R, M, beta, yR, k2, Lambda) or
    None on ANY failure -- never raises."""
    try:
        Pbar_min, Pbar_max = bundle["Pbar_min"], bundle["Pbar_max"]
        if not (Pbar_min <= Pbar_c <= Pbar_max):
            return None
        ebar_c = np.exp(bundle["logE_of_logP"](np.log(Pbar_c)))
        if not np.isfinite(ebar_c) or ebar_c <= 0:
            return None

        Rbar0 = 1e-6
        Mbar0 = ebar_c * Rbar0 ** 3 / 3
        y0 = [Mbar0, Pbar_c, 2.0]

        surface = lambda Rbar, y: _surface(Rbar, y, bundle)
        surface.terminal = True
        surface.direction = -1

        sol = solve_ivp(lambda Rb, y: _tov_rhs(Rb, y, bundle), (Rbar0, 5000), y0,
                         method="RK45", events=surface, rtol=1e-8, atol=1e-10)
        if not sol.success or len(sol.t_events[0]) == 0:
            return None

        Rbarf = sol.t_events[0][0]
        Mbarf, Pbarf, yRbar = sol.y_events[0][0]
        if not np.all(np.isfinite([Rbarf, Mbarf, yRbar])):
            return None

        R = Rbarf * R0
        beta = Mbarf / (2.0 * Rbarf)
        k2 = love_number_k2(beta, yRbar)
        if not np.isfinite(k2):
            return None
        lam = (2.0 / 3.0) * k2 * R ** 5
        M_geom_km = Mbarf * (R0 / 2.0)
        Lambda = lam / M_geom_km ** 5

        return {"Pbar_c": Pbar_c, "R": R, "M": Mbarf, "beta": beta,
                "yR": yRbar, "k2": k2, "lambda": lam, "Lambda": Lambda}
    except (ZeroDivisionError, ValueError, FloatingPointError):
        return None


# ---------------------------------------------------------------------------
# 3. forward_map: root-find to hit specific target masses instead of
#    sweeping the whole table
# ---------------------------------------------------------------------------
def _coarse_scan(bundle, n_coarse=40):
    """Coarse M(Pbar_c) scan, same log-spacing convention as the original
    3000-point sweep's bounds, just far fewer points. Used only to (a)
    locate Mmax's rough position and (b) bracket root-finds."""
    Pbar_min, Pbar_max = bundle["Pbar_min"], bundle["Pbar_max"]
    Pc_vals = np.logspace(np.log10(Pbar_min) + 2, np.log10(Pbar_max) - 0.1, n_coarse)
    out = []
    for Pc in Pc_vals:
        r = solve_star(Pc, bundle)
        if r is not None:
            out.append(r)
    return out


def _refine_mmax(bundle, coarse_results, half_window=2, n_local=15):
    """Local high-resolution scan around the coarse-scan's peak, then the
    same quadratic-vertex refinement as neutronstar_solver.refine_Mmax."""
    if len(coarse_results) < 3:
        return (coarse_results[-1]["M"] if coarse_results else None), None
    Pcs = np.array([r["Pbar_c"] for r in coarse_results])
    Ms = np.array([r["M"] for r in coarse_results])
    idx_max = int(np.argmax(Ms))

    lo = max(idx_max - half_window, 0)
    hi = min(idx_max + half_window + 1, len(Pcs))
    Pc_lo, Pc_hi = Pcs[max(lo, 0)], Pcs[min(hi, len(Pcs) - 1)]
    Pc_local = np.linspace(Pc_lo, Pc_hi, n_local)
    local_results = [solve_star(Pc, bundle) for Pc in Pc_local]
    local_results = [r for r in local_results if r is not None]
    if len(local_results) < 3:
        return float(Ms[idx_max]), coarse_results[idx_max]

    Pc_l = np.array([r["Pbar_c"] for r in local_results])
    M_l = np.array([r["M"] for r in local_results])
    a, b, c = np.polyfit(Pc_l, M_l, 2)
    best = local_results[int(np.argmax(M_l))]
    if a >= 0:
        return float(M_l.max()), best
    Pc_star = -b / (2 * a)
    M_star = a * Pc_star ** 2 + b * Pc_star + c
    return float(M_star), best


def forward_map(P_arr, eps_arr, target_masses, r_cut_km=50.0, n_coarse=40):
    """The main entry point for the likelihood.

    Parameters
    ----------
    P_arr, eps_arr : arrays, MeV/fm^3 -- one hybrid (or hadronic) EOS table
    target_masses : list of masses (Msun) to solve for directly, e.g.
                    [1.4, m1_gw, m2_gw, 2.07, 1.44, ...]
    r_cut_km : same low-density-branch guard as the original solver

    Returns
    -------
    dict with:
      'ok': bool -- overall success flag; if False, caller should treat the
            whole draw as log-likelihood = -inf and stop
      'reason': str, populated when ok=False
      'causal_ok': bool, 'max_cs2': float
      'Mmax': float or None
      'observables': {mass: {'R':.., 'Lambda':.., 'k2':.., 'beta':..}}
                     for each successfully-reached target mass (masses that
                     couldn't be reached on the stable branch are simply
                     absent from this dict -- caller's likelihood function
                     must handle a missing key as a rejection for whichever
                     data point needed that mass)
    """
    bundle = build_bundle(P_arr, eps_arr)
    if bundle is None:
        return {"ok": False, "reason": "invalid EOS table (non-monotonic or non-positive)"}

    causal_ok, max_cs2 = check_causality(bundle)
    if not causal_ok:
        return {"ok": False, "reason": f"causality/stability violated (max cs^2={max_cs2:.3f})",
                "causal_ok": False, "max_cs2": max_cs2}

    coarse = _coarse_scan(bundle, n_coarse=n_coarse)
    if len(coarse) < 5:
        return {"ok": False, "reason": "too few valid solves on coarse scan",
                "causal_ok": True, "max_cs2": max_cs2}

    Mmax, _ = _refine_mmax(bundle, coarse)
    if Mmax is None or not np.isfinite(Mmax):
        return {"ok": False, "reason": "Mmax refinement failed",
                "causal_ok": True, "max_cs2": max_cs2}

    # Stable-branch mass range from the coarse scan (same "walk back from
    # the peak while M is still increasing" logic as _finalize_results,
    # just applied to the coarse grid to get a search bracket).
    Pcs = np.array([r["Pbar_c"] for r in coarse])
    Ms = np.array([r["M"] for r in coarse])
    idx_max = int(np.argmax(Ms))
    start = idx_max
    while start > 0 and Ms[start - 1] < Ms[start]:
        start -= 1
    stable_Pc = Pcs[start:idx_max + 1]
    stable_M = Ms[start:idx_max + 1]

    if len(stable_M) < 2 or stable_M[-1] <= stable_M[0]:
        return {"ok": False, "reason": "no monotonic stable branch found on coarse scan",
                "causal_ok": True, "max_cs2": max_cs2, "Mmax": Mmax}

    observables = {}
    for M_target in target_masses:
        if not (stable_M[0] <= M_target <= stable_M[-1]):
            continue  # target unreachable on the stable branch -- left absent, not an error
        # bracket: find the coarse-grid interval straddling M_target
        j = np.searchsorted(stable_M, M_target)
        if j == 0 or j >= len(stable_M):
            continue
        Pc_lo, Pc_hi = stable_Pc[j - 1], stable_Pc[j]

        def f(Pc):
            r = solve_star(Pc, bundle)
            if r is None:
                return np.nan
            return r["M"] - M_target

        try:
            f_lo, f_hi = f(Pc_lo), f(Pc_hi)
            if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
                continue
            Pc_root = brentq(f, Pc_lo, Pc_hi, xtol=1e-10, rtol=1e-10, maxiter=100)
        except (ValueError, RuntimeError):
            continue

        r = solve_star(Pc_root, bundle)
        if r is None or r["R"] >= r_cut_km:
            continue
        observables[M_target] = {"R": r["R"], "Lambda": r["Lambda"], "k2": r["k2"], "beta": r["beta"]}

    return {"ok": True, "reason": "", "causal_ok": True, "max_cs2": max_cs2,
            "Mmax": Mmax, "observables": observables}


# ---------------------------------------------------------------------------
# 4. build_curve: a moderate-resolution R(M)/Lambda(M)/k2(M) interpolant.
#
# WHY THIS EXISTS ALONGSIDE forward_map(): forward_map's root-finding is the
# right tool when you need a SMALL, FIXED number of precise single-mass
# points (e.g. Stage D validation against the full 3000-point sweep, or a
# quick R_1.4 check) -- each extra target mass costs one more brentq bracket
# (a handful of solves).
#
# But the likelihood the mentee specified needs DENSE marginalization: each
# NICER source's 2D KDE has to be evaluated along the whole (M, R(M)) curve
# over that source's mass support (not just at one mass), and GW170817's
# Lambda1-Lambda2 KDE has to be evaluated at (Lambda1(m1), Lambda2(m2)) for
# ~O(30-100) mass-pairs per likelihood call to Monte-Carlo-marginalize over
# the released (m1, m2) posterior. Repeating forward_map's root-find for
# every one of those masses would mean O(100s) of brentq brackets PER
# likelihood call -- far more solve_ivp calls than just building one
# moderate-resolution curve (~80-150 points, vs. the original's 3000) and
# reading everything off a PchipInterpolator, which is free after the scan.
#
# So: use build_curve() inside the actual Bayesian likelihood (likelihood.py)
# and reserve forward_map()'s root-finding for validate_forward_map.py's
# precise single-point checks.
# ---------------------------------------------------------------------------
def build_curve(P_arr, eps_arr, n_scan=100, r_cut_km=50.0):
    bundle = build_bundle(P_arr, eps_arr)
    if bundle is None:
        return {"ok": False, "reason": "invalid EOS table"}

    causal_ok, max_cs2 = check_causality(bundle)
    if not causal_ok:
        return {"ok": False, "reason": f"acausal (max cs^2={max_cs2:.3f})",
                "causal_ok": False, "max_cs2": max_cs2}

    Pbar_min, Pbar_max = bundle["Pbar_min"], bundle["Pbar_max"]
    Pc_vals = np.logspace(np.log10(Pbar_min) + 2, np.log10(Pbar_max) - 0.1, n_scan)
    results = [solve_star(Pc, bundle) for Pc in Pc_vals]
    results = [r for r in results if r is not None and r["R"] < r_cut_km]
    if len(results) < 8:
        return {"ok": False, "reason": "too few valid solves on scan",
                "causal_ok": True, "max_cs2": max_cs2}

    Pcs = np.array([r["Pbar_c"] for r in results])
    Ms = np.array([r["M"] for r in results])
    idx_max = int(np.argmax(Ms))

    Mmax, _ = _refine_mmax(bundle, results)

    start = idx_max
    while start > 0 and Ms[start - 1] < Ms[start]:
        start -= 1
    s = slice(start, idx_max + 1)
    if (s.stop - s.start) < 3:
        return {"ok": False, "reason": "stable branch too short to interpolate",
                "causal_ok": True, "max_cs2": max_cs2, "Mmax": Mmax}

    M_s = Ms[s]
    order = np.argsort(M_s)
    M_sorted = M_s[order]
    Rs = np.array([r["R"] for r in results])[s][order]
    Lambdas = np.array([r["Lambda"] for r in results])[s][order]
    k2s = np.array([r["k2"] for r in results])[s][order]

    return {
        "ok": True, "causal_ok": True, "max_cs2": max_cs2, "Mmax": Mmax,
        "M_range": (float(M_sorted[0]), float(M_sorted[-1])),
        "R_of_M": PchipInterpolator(M_sorted, Rs, extrapolate=False),
        "Lambda_of_M": PchipInterpolator(M_sorted, Lambdas, extrapolate=False),
        "k2_of_M": PchipInterpolator(M_sorted, k2s, extrapolate=False),
    }
