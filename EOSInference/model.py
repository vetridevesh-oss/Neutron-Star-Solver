"""
model.py -- ties prior, EOS, solver and likelihood into the callables dynesty needs.

Kept in its own module so that multiprocessing workers can unpickle it.

Every hard constraint lives in HybridModel.screen(), and nowhere else.  Both
evaluate() (what dynesty integrates) and acceptance.classify() (what measures
how much of the prior survives, and hence the evidence renormalisation) call it,
so the measured acceptance fraction cannot drift away from the filters the
sampler actually sees.
"""
import numpy as np

import config
import solver
from hybrid_eos import HadronicTable, HybridEOS
from priors import PARAM_NAMES, PriorTransform

# Outcomes of HybridModel.screen(), in the order they are tested.  Only "accepted"
# earns a likelihood.  "solver_failure" is a numerical outcome; the other four are
# the prior truncation whose volume acceptance.py measures.
SCREEN_KINDS = ("invalid", "acausal_base_poly", "acausal_hadronic",
                "solver_failure", "acausal_in_star", "accepted")
PRIOR_REJECTIONS = ("invalid", "acausal_base_poly", "acausal_hadronic", "acausal_in_star")


class HybridModel:
    def __init__(self, eos_file, likelihood, sweep_cfg=None, prior=None):
        self.eos_file = eos_file
        self.table = HadronicTable(eos_file, config.EOS_SKIPROWS, config.EOS_COLS)
        self.like = likelihood
        self.sweep_cfg = sweep_cfg or config.SWEEP
        self.prior_transform = PriorTransform(self.table, prior)
        self.ndim = len(PARAM_NAMES)

    # ---- screening (the single definition of the hard constraints) ----
    def screen(self, theta):
        """Apply every hard constraint to one physical parameter vector.

        Returns (kind, eos, seq, reason) with `kind` one of SCREEN_KINDS.  "accepted"
        means the EOS passed every cut and `seq` is usable; anything else means the
        draw is outside the model's support and gets ln L = -inf.  "solver_failure"
        is a numerical outcome rather than a prior truncation, which is why it is a
        kind of its own (see acceptance.py).

        This is the only place the constraints are written down: evaluate() below
        and acceptance.classify() both go through it.
        """
        tol = self.sweep_cfg["causality_tol"]
        eos = HybridEOS(self.table, *theta)
        if not eos.valid:
            return "invalid", eos, None, eos.reason
        # cheap causality screens before any TOV integration
        if theta[2] * eos.P_t / eos.eps_s > 1 + tol:
            return "acausal_base_poly", eos, None, "cs2>1 at base of the polytrope"
        if self.table.cs2_max_below(eos.P_t) > 1 + tol:
            return "acausal_hadronic", eos, None, "hadronic cs2>1 below eps_trans"
        seq = solver.compute_sequence(eos, self.sweep_cfg)
        if not seq.ok:
            return "solver_failure", eos, seq, seq.reason
        if seq.cs2_max > 1 + tol:
            return ("acausal_in_star", eos, seq,
                    "acausal inside stable stars (cs2=%.3f)" % seq.cs2_max)
        return "accepted", eos, seq, ""

    # ---- evaluation ----
    def evaluate(self, theta):
        """Returns dict(logL, seq, eos, reason, kind). logL = -inf for rejected EOS."""
        kind, eos, seq, reason = self.screen(theta)
        out = dict(logL=-np.inf, seq=seq, eos=eos, reason=reason, kind=kind)
        if kind == "accepted":
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
