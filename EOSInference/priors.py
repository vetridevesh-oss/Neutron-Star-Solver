"""
priors.py -- unit-cube -> physical parameters for the three-parameter hybrid core.

Sampled vector (what dynesty sees and what the posterior samples contain):
    theta = [eps_trans, delta_eps_trans, gamma]
energy densities in MeV/fm^3.  All ranges come from config.PRIOR.

    eps_trans        ~ U[lo, hi]                         (lo, hi per hadronic table)
    delta_eps_trans  ~ eps_trans * U[frac range]         ("fractional" mode)
                     or U[abs range]                     ("absolute" mode)
    gamma            ~ U[range]

NOTE: the fractional construction is a prior on the physical vector that is *not*
uniform in the absolute delta_eps; Bayes factors depend on that choice
(config.PRIOR["delta_mode"]).
"""
import numpy as np

import config

PARAM_NAMES = ["eps_trans", "delta_eps_trans", "gamma"]
PARAM_LABELS = [r"$\varepsilon_{\rm trans}$", r"$\Delta\varepsilon_{\rm trans}$", r"$\gamma$"]


class PriorTransform:
    def __init__(self, table, prior=None):
        self.p = prior or config.PRIOR
        self.eps_lo, self.eps_hi = table.eps_range_from_prior(self.p["eps_trans"])
        if self.p["delta_mode"] not in ("fractional", "absolute"):
            raise ValueError("PRIOR['delta_mode'] must be 'fractional' or 'absolute'")

    def __call__(self, u):
        p = self.p
        eps_t = self.eps_lo + u[0] * (self.eps_hi - self.eps_lo)
        lo, hi = p["frac_delta_trans"] if p["delta_mode"] == "fractional" else p["abs_delta_trans"]
        d_t = (eps_t if p["delta_mode"] == "fractional" else 1.0) * (lo + u[1] * (hi - lo))
        g = p["gamma"][0] + u[2] * (p["gamma"][1] - p["gamma"][0])
        return np.array([eps_t, d_t, g])
