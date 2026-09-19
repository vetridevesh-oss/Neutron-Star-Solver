"""
likelihood.py -- combines the per-source terms into one log-likelihood for
a hybrid-EOS parameter draw theta.

Structure (log-additive across independent sources):
    logL(theta) = logL_J0030 + logL_J0740 + logL_J0437 + logL_GW170817 + logL_J0952

NICER terms: for each source, Monte-Carlo/grid-marginalize the source's own
2D (M,R) KDE over M along the EOS's predicted R(M) curve -- NOT a single
point evaluation at the source's mean mass, because the "mass" of a NICER
source is itself part of what that source's own posterior constrains (it is
not independently, precisely known the way a radio-timing mass is).

GW170817 term: per your methodological note, this bypasses a compressed
Lambda_1.4 value entirely. For each of a FIXED (chosen once per run, not
resampled every likelihood call -- see note below) subset of the released
GW170817 (m1, m2) posterior samples, compute (Lambda1(m1), Lambda2(m2)) from
the EOS's own Lambda(M) curve and evaluate the released Lambda1-Lambda2 KDE
there, then average (Monte Carlo marginalization over the mass posterior).

WHY THE GW MASS SUBSAMPLE IS FIXED PER RUN, NOT PER CALL: nested sampling's
evidence integral assumes a deterministic (theta -> logL) map. If a
different random subset of (m1,m2) pairs were drawn on every likelihood
call, logL(theta) would have sampling noise INDEPENDENT of theta, which
biases/slows dynesty's exploration (it can mistake noise for structure).
Fix the subsample once (via a seeded RNG) before the nested-sampling run
starts, and reuse the same subsample for every call in that run.
"""

import numpy as np
from scipy.special import logsumexp

FLOOR = -50.0  # log-likelihood floor for "technically in range but essentially zero support"


def _nicer_term(curve, source, n_mass_grid=25):
    if not curve["ok"]:
        return -np.inf
    lo = max(curve["M_range"][0], source.x_range[0])
    hi = min(curve["M_range"][1], source.x_range[1])
    if lo >= hi:
        return -np.inf  # EOS's stable branch never overlaps this source's mass support at all
    M_grid = np.linspace(lo, hi, n_mass_grid)
    R_grid = curve["R_of_M"](M_grid)
    valid = np.isfinite(R_grid)
    if not np.any(valid):
        return -np.inf
    vals = np.array([source.log_prob(M, R) for M, R in zip(M_grid[valid], R_grid[valid])])
    return float(logsumexp(vals) - np.log(len(vals)))


def _gw170817_term(curve, gw_kde, gw_mass_subsample):
    if not curve["ok"]:
        return -np.inf
    lo, hi = curve["M_range"]
    m1, m2 = gw_mass_subsample[:, 0], gw_mass_subsample[:, 1]
    in_range = (m1 >= lo) & (m1 <= hi) & (m2 >= lo) & (m2 <= hi)
    if not np.any(in_range):
        return -np.inf  # EOS can't produce stars at the merger's component masses at all
    Lambda1 = curve["Lambda_of_M"](m1[in_range])
    Lambda2 = curve["Lambda_of_M"](m2[in_range])
    valid = np.isfinite(Lambda1) & np.isfinite(Lambda2)
    if not np.any(valid):
        return -np.inf
    vals = np.array([gw_kde.log_prob(L1, L2) for L1, L2 in zip(Lambda1[valid], Lambda2[valid])])
    return float(logsumexp(vals) - np.log(len(vals)))


def _select_terms(sources, nicer_scenario, n_gw_subsample, rng_seed):
    j0030 = sources["J0030_C"] if nicer_scenario == "C" else sources["J0030_B"]
    nicer_terms = [j0030, sources["J0740"], sources["J0437"]]
    rng = np.random.default_rng(rng_seed)
    all_gw_masses = sources["GW170817_masses"]
    idx = rng.choice(len(all_gw_masses), size=min(n_gw_subsample, len(all_gw_masses)), replace=False)
    gw_mass_subsample = all_gw_masses[idx]
    return nicer_terms, sources["GW170817_Lambda"], gw_mass_subsample, sources["J0952"]


def _full_loglike_from_curve(curve, nicer_terms, gw_kde, gw_mass_subsample, j0952, n_mass_grid=25):
    if not curve["ok"]:
        return -np.inf
    logL = 0.0
    for source in nicer_terms:
        logL += _nicer_term(curve, source, n_mass_grid=n_mass_grid)
        if not np.isfinite(logL):
            return -np.inf
    logL += _gw170817_term(curve, gw_kde, gw_mass_subsample)
    if not np.isfinite(logL):
        return -np.inf
    logL += j0952.log_prob(curve["Mmax"])
    return float(logL)


def make_log_likelihood(hadronic, sources, nicer_scenario="C", n_gw_subsample=32,
                         n_mass_grid=25, rng_seed=42, n_scan=100):
    """Returns a log_likelihood(theta_array) closure for dynesty -- the
    HYBRID (6-parameter) model's likelihood.

    nicer_scenario: "C" (primary) or "B" (ablation) -- selects which J0030
    KDE is used; a separate nested-sampling run is done per scenario
    (per your resolved decision), not a mixture within one run.
    """
    import hybrid_eos
    import solver_interface
    from priors import theta_array_to_dict

    nicer_terms, gw_kde, gw_mass_subsample, j0952 = _select_terms(
        sources, nicer_scenario, n_gw_subsample, rng_seed)

    def log_likelihood(theta_array):
        theta = theta_array_to_dict(theta_array)
        table = hybrid_eos.build_hybrid_eos(hadronic, theta)
        if not table["valid"]:
            return -np.inf
        curve = solver_interface.build_curve(table["P"], table["eps"], n_scan=n_scan)
        return _full_loglike_from_curve(curve, nicer_terms, gw_kde, gw_mass_subsample, j0952, n_mass_grid)

    return log_likelihood


def hadronic_null_loglike(hadronic, sources, nicer_scenario="C", n_gw_subsample=32,
                           n_mass_grid=25, rng_seed=42, n_scan=100):
    """The ZERO-parameter hadronic model's evidence. Per the Bayes-factor
    asymmetry noted earlier in this project: a fixed EOS with no free
    parameters needs no nested sampling at all -- its evidence IS this
    single log-likelihood value, Z_had = L(fixed hadronic EOS)."""
    import solver_interface

    nicer_terms, gw_kde, gw_mass_subsample, j0952 = _select_terms(
        sources, nicer_scenario, n_gw_subsample, rng_seed)

    P_grid = np.logspace(np.log10(hadronic.P_min), np.log10(hadronic.P_max), 400)
    eps_grid = hadronic.eps(P_grid)
    curve = solver_interface.build_curve(P_grid, eps_grid, n_scan=n_scan)
    return _full_loglike_from_curve(curve, nicer_terms, gw_kde, gw_mass_subsample, j0952, n_mass_grid)
