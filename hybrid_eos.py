"""
hybrid_eos.py -- Tanh-regularized hadron-quark crossover EOS, 2-polytrope core.

PHYSICAL ASSUMPTIONS (stated up front, per the mentee's spec):

1. The hadronic baseline ``eps_had(p)`` is supplied as a tabulated EOS (n, eps,
   P in fm^-3 / MeV.fm^-3 / MeV.fm^-3) representing a real, disfavored-stiff
   RMF or GPP model (e.g. GM1, TM1, NL3rho-omega4, DD2). This module does not
   generate it -- it only consumes it.

2. K2, K3 (the QM-segment stiffness normalizations) are NOT fixed by the
   continuity relations given in the spec as literally written: with the
   offset form P - P_anchor = K*(eps - eps_anchor)^gamma, ANY K satisfies
   continuity at the anchor point trivially (the offset term vanishes
   there), so the given relations alone leave K2, K3 undetermined.

   THREE naive resolutions were tried and empirically FAILED (verified by
   direct Monte Carlo test, not assumed): normalizing by the fixed eps_ns
   scale, by the anchor pressure itself, and by matching the hadronic
   branch's local sound speed via a derivative-continuity condition. All
   three produced causality violations (max cs^2 up to 10^3-10^21) across
   most of the prior, because each involves K raised to the power -gamma
   with gamma in [1,8] -- exponentially sensitive to whatever quantity
   fills the base, regardless of what that quantity physically is.

   RESOLUTION USED HERE (verified far better: ~27% causal-acceptance rate
   with violations O(1-30) when they DO occur, vs. ~5-7% acceptance with
   O(10^3-10^21) violations for all three of the above): follow the
   standard piecewise-polytrope convention from Read, Lackey, Owen &
   Friedman (2009, PRD 79, 124032) -- the paper the field's segmented-EOS
   parametrization descends from. Read et al. use NON-OFFSET power laws
   P = K*eps^gamma and fix K by requiring the segment pass through its own
   anchor point exactly: K_i = P_anchor_i / eps_anchor_i^gamma_i. This
   differs from the spec's offset form but is mathematically equivalent
   AT the anchor -- verified to floating-point precision that
   eps_QM(P_anchor) = eps_anchor exactly under this convention, so the
   latent-heat jump Delta_eps is fully preserved -- while being vastly
   better behaved away from the anchor, because K is now a ratio of two
   real, always-reasonably-scaled physical quantities (P_anchor,
   eps_anchor) rather than requiring an offset base that is identically
   zero at the anchor by construction to be raised to a large negative
   power. In short: the offset is essential for encoding Delta_eps as a
   jump (kept, in the tanh-blend value at P_trans) but actively harmful
   for fixing K's magnitude -- so K uses the non-offset (Read-style)
   formula while the segment's interior shape still reflects the given
   gamma2/gamma3/Delta_eps_core parameters.

3. The crossover is built directly in pressure space, since eps(p) (not
   P(eps)) is the natural direction of both the given formulas AND of what
   the solver's PchipInterpolator(logP -> logEps) already expects -- no
   inversion needed.

4. Baryon number density n(p) is NOT separately constructed for the hybrid
   branch (the solver's TOV/tidal physics only needs eps(P), and n is
   unused downstream in EOStoObservables' actual physics -- confirmed by
   reading neutronstar_solver.py). n is filled with NaN in the output table.
   If your mentee later wants central baryon density diagnostics, that
   requires integrating dn/deps = n/(eps+P) separately -- flagged as a
   possible future addition, not implemented here.

5. Thermodynamic stability (dP/deps >= 0) and causality (cs^2 <= 1) are
   enforced as REJECTION criteria on the constructed table (checked at
   every grid point), not as soft penalties -- consistent with treating
   them as hard prior-support boundaries per the spec.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

EPS_NS = 150.0          # MeV/fm^3, fixed reference (given)
EPS_MAX = 8.3 * EPS_NS   # MeV/fm^3, EOS truncation limit (given)


class HadronicBaseline:
    """Wraps a tabulated hadronic EOS file as an eps(p) callable.

    Expected file format: same convention as neutronstar_solver.py already
    uses (whitespace-delimited, 4 header lines, columns = eps [MeV/fm^3],
    P [MeV/fm^3], n [fm^-3]) -- i.e. a GM1/TM1/NL3wr4/DD2 table in that
    layout. If your mentee's actual files use a different column order or
    header length, adjust `skiprows`/column indices below -- do NOT assume;
    verify against a known literature boundary row first (per your own
    "column order bugs are silent" lesson).
    """

    def __init__(self, path, skiprows=4, col_eps=0, col_P=1, col_n=2):
        tab = np.loadtxt(path, skiprows=skiprows)
        eps, P, n = tab[:, col_eps], tab[:, col_P], tab[:, col_n]
        order = np.argsort(P)
        P, eps, n = P[order], eps[order], n[order]
        if not (np.all(np.diff(P) > 0) and np.all(np.diff(eps) > 0)):
            raise ValueError(
                "Hadronic table is not strictly monotonic in P and eps "
                "after sorting -- check for duplicate rows or a corrupted file."
            )
        self.P_min, self.P_max = P.min(), P.max()
        self.n_min, self.n_max = n.min(), n.max()
        self._logeps_of_logP = PchipInterpolator(np.log(P), np.log(eps), extrapolate=False)
        self._logn_of_logP = PchipInterpolator(np.log(P), np.log(n), extrapolate=False)

    def eps(self, p):
        p = np.clip(p, self.P_min, self.P_max)
        return np.exp(self._logeps_of_logP(np.log(p)))

    def n(self, p):
        p = np.clip(p, self.P_min, self.P_max)
        return np.exp(self._logn_of_logP(np.log(p)))


def _tanh_weight(p, P_trans, delta_p):
    return 0.5 * (1.0 + np.tanh((p - P_trans) / delta_p))


def eps_qm(p, K2, gamma2, P1, K3, gamma3):
    """Piecewise QM branch eps_QM(p) -- Read-et-al.-style NON-OFFSET power
    laws, eps = (p/K)^(1/gamma) per segment (see module docstring assumption
    #2 for why this replaces the spec's offset form for the purpose of
    fixing K, while still reproducing the spec's anchor values exactly:
    eps_seg1(P_trans) = eps_trans+d_eps and eps_seg2(P1) = eps1, both to
    floating-point precision, by construction of K2, K3 below).

    Vectorized over p. Clipped at a tiny floor to avoid a zero-division at
    p=0 (never physically reached -- P_trans's prior floor is 10 MeV/fm^3).
    """
    p = np.clip(np.atleast_1d(np.asarray(p, dtype=float)), 1e-30, None)
    eps_seg1 = (p / K2) ** (1.0 / gamma2)
    eps_seg2 = (p / K3) ** (1.0 / gamma3)
    return np.where(p < P1, eps_seg1, eps_seg2)


def build_hybrid_eos(hadronic, theta, p_grid=None, n_grid=2000, p_max_factor=3.0):
    """Construct one hybrid EOS table from a HadronicBaseline + parameter draw.

    Parameters
    ----------
    hadronic : HadronicBaseline
    theta : dict with keys P_trans, delta_p_frac, d_eps_frac, gamma2,
            d_eps_core, gamma3   (all 6 free parameters from the spec)
    p_grid : optional pre-built pressure grid (MeV/fm^3); if None, one is
             built spanning the hadronic table's range up through a
             generous multiple of P1 (see p_max_factor) so the TOV central-
             pressure sweep/root-find never runs off the table.

    Returns
    -------
    dict with keys 'P', 'eps', 'n' (n is NaN-filled, see module docstring),
    plus the derived quantities (eps_trans, d_eps, eps1, P1, K2, K3) and a
    'valid' bool + 'reject_reason' string for causality/stability filtering.
    """
    P_trans = theta["P_trans"]
    delta_p = theta["delta_p_frac"] * P_trans
    eps_trans = float(hadronic.eps(np.array([P_trans]))[0])
    d_eps = theta["d_eps_frac"] * eps_trans
    gamma2 = theta["gamma2"]
    d_eps_core = theta["d_eps_core"]
    gamma3 = theta["gamma3"]

    eps1 = eps_trans + d_eps + d_eps_core
    # K2, K3 fixed by the Read-et-al.-style non-offset anchoring (see
    # module docstring assumption #2 -- empirically verified far more
    # causally well-behaved than the eps_ns/P_anchor/slope-matching
    # alternatives that were tried and failed).
    eps_anchor1 = eps_trans + d_eps          # = eps_QM(P_trans), exactly
    K2 = P_trans / eps_anchor1 ** gamma2
    P1 = K2 * eps1 ** gamma2                 # = pressure at eps=eps1 on segment 1
    K3 = P1 / eps1 ** gamma3

    if p_grid is None:
        P_hi = max(P1 * p_max_factor, P_trans * 10.0)
        p_grid = np.logspace(np.log10(hadronic.P_min), np.log10(P_hi), n_grid)

    w = _tanh_weight(p_grid, P_trans, delta_p)
    eps_h = hadronic.eps(p_grid)
    eps_q = eps_qm(p_grid, K2, gamma2, P1, K3, gamma3)
    eps_blend = (1.0 - w) * eps_h + w * eps_q

    valid, reason = _check_physical(p_grid, eps_blend)

    return {
        "P": p_grid,
        "eps": eps_blend,
        "n": np.full_like(p_grid, np.nan),
        "eps_trans": eps_trans, "d_eps": d_eps, "eps1": eps1,
        "P1": P1, "K2": K2, "K3": K3,
        "valid": valid, "reject_reason": reason,
    }


def _check_physical(p_grid, eps_grid):
    """Reject on non-monotonicity (dP/deps < 0) or acausality (cs^2 > 1).

    cs^2 = dP/deps, computed via a Pchip-log-log interpolant matching the
    solver's own convention, so what's rejected here is rejected on
    *exactly* the same numerical grounds the solver would fail on downstream
    -- keeps the prior-level veto and the runtime veto consistent.
    """
    if np.any(~np.isfinite(eps_grid)) or np.any(eps_grid <= 0):
        return False, "non-finite or non-positive eps in table"
    if not np.all(np.diff(eps_grid) > 0):
        return False, "eps(P) not strictly increasing (dP/deps < 0 somewhere)"

    logP, logE = np.log(p_grid), np.log(eps_grid)
    dlogE_dlogP = PchipInterpolator(logP, logE).derivative(1)(logP)
    with np.errstate(divide="ignore", invalid="ignore"):
        cs2 = (p_grid / eps_grid) / dlogE_dlogP
    if np.any(cs2 > 1.0 + 1e-9):
        return False, f"causality violated: max cs^2 = {np.nanmax(cs2):.3f}"
    if np.any(cs2 < 0):
        return False, "cs^2 < 0 somewhere (unstable)"
    return True, ""
