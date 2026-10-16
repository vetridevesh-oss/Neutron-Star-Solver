"""
mockdata.py -- synthetic data drawn around a known "true" stellar sequence.

For pipeline tests and injection-recovery only.  The synthetic sets are built in
exactly the variables the real likelihood consumes -- (M, R) for the pulsars and
(q, Lambda1, Lambda2) at fixed chirp mass for the binary -- and are handed to
`Likelihood` as ready-made terms, so a mock run exercises the same code path as
a real one.
"""
import numpy as np

import config
import datasets as D
from likelihood import GWTerm, Likelihood, MassRadiusTerm


def _true_value(seq, m, key):
    """Value of `key` at mass m on the most massive branch that contains it."""
    v = seq.branch_values(np.atleast_1d(m), key)
    with np.errstate(invalid="ignore"):
        return np.nanmax(v, axis=0)


def make_mock_likelihood(seq, seed=0, n=20000, like_cfg=None, sources=None, mc=1.1858):
    """Synthetic NICER-like and GW-like data around `seq`.

    `sources` is a list of (name, m0, sigma_M, sigma_R) in solar masses / km.
    """
    lc = dict(like_cfg or config.LIKELIHOOD)
    rng = np.random.default_rng(seed)
    sources = sources or [("J0030", 1.34, 0.15, 0.9), ("J0740", 2.07, 0.07, 1.0),
                          ("J0437", 1.42, 0.06, 0.7)]
    terms = []
    for name, m0, sm, sr in sources:
        m0 = min(m0, seq.M_tov - 0.05)
        r0 = float(_true_value(seq, m0, "R"))
        m = rng.normal(m0, sm, n)
        r = rng.normal(r0, sr, n)
        x = np.column_stack([m, r])
        ss = D.SampleSet(x, None, name, "<mock>", {"M": 0, "R": 1}, note="synthetic")
        kde = D.GridKDE(ss.x, ss.w, bw_factor=lc["bw_factor_mr"])
        terms.append(MassRadiusTerm(name, ss, kde, lc["mass_prior_min"], lc["n_mass_grid"]))

    # GW: draw q from the same q^(-6/5)(1+q)^(2/5) shape the real sampling prior has,
    # so that a mock run exercises the same prior division
    qg = np.linspace(0.55, 1.0, 2000)
    pq = qg ** -1.2 * (1 + qg) ** 0.4
    cq = np.cumsum(pq)
    cq /= cq[-1]
    q = np.interp(rng.uniform(size=n), cq, qg)
    m1, m2 = D.masses_from_Mc_q(mc, q)
    L1, L2 = _true_value(seq, m1, "Lam"), _true_value(seq, m2, "Lam")
    ok = np.isfinite(L1) & np.isfinite(L2)
    L1 = np.clip(L1[ok] * rng.lognormal(0, 0.35, ok.sum()), 0, 5000)
    L2 = np.clip(L2[ok] * rng.lognormal(0, 0.35, ok.sum()), 0, 5000)
    gss = D.SampleSet(np.column_stack([q[ok], L1, L2]), None, "GWmock", "<mock>",
                      {"q": 0, "Lambda1": 1, "Lambda2": 2}, note="synthetic")
    gss.Mc = mc
    gkde = D.GridKDE(gss.x, gss.w, bw_factor=lc["bw_factor_gw"],
                     bounds=((0, 1.0, "hi"), (1, 0.0, "lo"), (2, 0.0, "lo")))
    gw = GWTerm("GWmock", gss, gkde, mc, lc["mass_prior_min"], lc["n_q_grid"],
                lc["gw_population"], lc["q_min"])
    return Likelihood(mr_terms=terms, gw_term=gw, mass_floor=config.DATA["mass_floor"],
                      like_cfg=lc)
