"""
priors.py -- unit-cube -> physical parameters for the six-parameter hybrid core.

Sampled vector (what dynesty sees and what the posterior samples contain):
    theta = [eps_trans, delta_eps_trans, eps_1, delta_eps_1, gamma1, gamma2]
all energy densities in MeV/fm^3.  All ranges come from config.PRIOR.

Hierarchy (every draw satisfies eps_1 > eps_trans + delta_eps_trans by construction):
    eps_trans              ~ U[lo, hi]
    delta_eps_trans        ~ eps_trans * U[frac range]   (fractional mode)  or U[abs range]
    start1 = eps_trans + delta_eps_trans
    eps_1 - start1         ~ U[width range]                                 ["width" mode]
                          or U[max(start1+gap, abs_lo), abs_hi]            ["absolute" mode]
                          truncated above at PRIOR["eps1_cap"]; draws with no room
                          left below the cap fall outside the support
    delta_eps_1            ~ eps_1 * U[frac range]       or U[abs range]
    gamma1, gamma2         ~ U[range]

NOTE: the fractional and conditional constructions are priors on the physical
vector that are *not* uniform in the absolute delta_eps / eps_1; Bayes factors
depend on that choice (config.PRIOR["delta_mode"], ["eps1_mode"]).
"""
import numpy as np

import config

PARAM_NAMES = ["eps_trans", "delta_eps_trans", "eps_1", "delta_eps_1", "gamma1", "gamma2"]
PARAM_LABELS = [r"$\varepsilon_{\rm trans}$", r"$\Delta\varepsilon_{\rm trans}$", r"$\varepsilon_1$",
                r"$\Delta\varepsilon_1$", r"$\gamma_1$", r"$\gamma_2$"]


class PriorTransform:
    def __init__(self, table, prior=None):
        self.p = prior or config.PRIOR
        self.eps_lo, self.eps_hi = table.eps_range_from_prior(self.p["eps_trans"])
        if self.p["delta_mode"] not in ("fractional", "absolute"):
            raise ValueError("PRIOR['delta_mode'] must be 'fractional' or 'absolute'")
        if self.p["eps1_mode"] not in ("width", "absolute"):
            raise ValueError("PRIOR['eps1_mode'] must be 'width' or 'absolute'")

    def __call__(self, u):
        p = self.p
        eps_t = self.eps_lo + u[0] * (self.eps_hi - self.eps_lo)
        if p["delta_mode"] == "fractional":
            lo, hi = p["frac_delta_trans"]
            d_t = eps_t * (lo + u[1] * (hi - lo))
        else:
            lo, hi = p["abs_delta_trans"]
            d_t = lo + u[1] * (hi - lo)
        start1 = eps_t + d_t
        cap = p.get("eps1_cap")
        if p["eps1_mode"] == "width":
            wlo, whi = p["eps1_width"]
            # the width of polytrope 1 (eps_1 - start1) is uniform in the width range
            lower, upper = start1 + wlo, start1 + whi
        else:
            alo, ahi = p["eps1_abs"]
            lower, upper = max(start1 + p["eps1_min_gap"], alo), ahi
        if cap is not None:
            upper = min(upper, cap)
        # upper <= lower means this (eps_trans, delta_eps_trans) leaves no room for a
        # second transition below the cap.  Return a point the EOS constructor
        # rejects, so the draw contributes zero to the evidence integral: that is
        # exactly what a truncated prior should do, and it keeps the map from the
        # unit cube total (dynesty requires that).
        eps_1 = lower + u[2] * (upper - lower) if upper > lower else start1
        if p["delta_mode"] == "fractional":
            lo, hi = p["frac_delta_1"]
            d_1 = eps_1 * (lo + u[3] * (hi - lo))
        else:
            lo, hi = p["abs_delta_1"]
            d_1 = lo + u[3] * (hi - lo)
        g1 = p["gamma1"][0] + u[4] * (p["gamma1"][1] - p["gamma1"][0])
        g2 = p["gamma2"][0] + u[5] * (p["gamma2"][1] - p["gamma2"][0])
        return np.array([eps_t, d_t, eps_1, d_1, g1, g2])
