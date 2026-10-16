"""
model.py -- ties prior, EOS, solver and likelihood into the callables dynesty needs.

Kept in its own module so that multiprocessing workers can unpickle it.
"""
import numpy as np

import config
import solver
from hybrid_eos import HadronicTable, HybridEOS
from priors import PriorTransform


class HybridModel:
    def __init__(self, eos_file, likelihood, sweep_cfg=None, prior=None):
        self.eos_file = eos_file
        self.table = HadronicTable(eos_file, config.EOS_SKIPROWS, config.EOS_COLS)
        self.like = likelihood
        self.sweep_cfg = sweep_cfg or config.SWEEP
        self.prior_transform = PriorTransform(self.table, prior)
        self.ndim = 6

    # ---- evaluation ----
    def evaluate(self, theta):
        """Returns dict(logL, seq, eos, reason). logL = -inf for rejected EOS."""
        eos = HybridEOS(self.table, *theta)
        out = dict(logL=-np.inf, seq=None, eos=eos, reason="")
        if not eos.valid:
            out["reason"] = eos.reason
            return out
        # cheap causality screen before any TOV integration
        g1 = theta[4]
        if g1 * eos.P_t / eos.eps_s1 > 1 + self.sweep_cfg["causality_tol"]:
            out["reason"] = "cs2>1 at base of polytrope 1"
            return out
        if self.table.cs2_max_below(eos.P_t) > 1 + self.sweep_cfg["causality_tol"]:
            out["reason"] = "hadronic cs2>1 below eps_trans"
            return out
        seq = solver.compute_sequence(eos, self.sweep_cfg)
        out["seq"] = seq
        if not seq.ok:
            out["reason"] = seq.reason
            return out
        if seq.cs2_max > 1 + self.sweep_cfg["causality_tol"]:
            out["reason"] = "acausal inside stable stars (cs2=%.3f)" % seq.cs2_max
            return out
        out["logL"] = self.like(seq)
        return out

    def loglike(self, theta):
        return self.evaluate(theta)["logL"]

    def hadronic_baseline(self):
        """ln L of the pure hadronic EOS (no free parameters, so Z_had = L_had)."""
        eos = HybridEOS.pure_hadronic(self.table)
        seq = solver.compute_sequence(eos, self.sweep_cfg)
        causal = eos.max_cs2_up_to(seq.Pc_max_stable) <= 1 + self.sweep_cfg["causality_tol"]
        return (self.like(seq) if (seq.ok and causal) else -np.inf), seq
