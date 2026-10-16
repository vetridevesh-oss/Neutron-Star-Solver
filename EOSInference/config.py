"""
config.py -- every setting you are likely to want to change lives here.

Nothing else in the package hard-codes a prior range, a file path, a
tolerance or a likelihood choice.  Edit this file (or override entries from
the command line / a notebook:  `import config; config.PRIOR["gamma1"] = (1, 6)`).

Units: energy density and pressure in MeV/fm^3, number density in fm^-3,
masses in solar masses, radii in km.
"""

N0 = 0.16  # nuclear saturation density [fm^-3], used only to express density bounds

# --------------------------------------------------------------------------
# 1. Hadronic EOS table(s)  (crust included).  Columns read from the file:
#    EOS_COLS = (energy density, pressure, number density), as in your solver.
# --------------------------------------------------------------------------
# Paths are relative to this directory (the tables live in the repository root).
HADRONIC_EOS_FILES = ["../EOSBetaFSU2L80.dat"]
EOS_SKIPROWS = 4
EOS_COLS = (0, 1, 2)

# --------------------------------------------------------------------------
# 2. PRIOR RANGES  (uniform priors; the six sampled parameters are
#    eps_trans, delta_eps_trans, eps_1, delta_eps_1, gamma1, gamma2)
#
# eps_trans : energy density at the top of the hadronic phase (pre-jump side)
#     kind = "n0"      -> bounds given in units of n0, converted to energy
#                         density with the hadronic table's own n(eps) column
#     kind = "MeV/fm3" -> bounds are energy densities directly
# --------------------------------------------------------------------------
PRIOR = {
    "eps_trans": {"kind": "n0", "lo": 1.0, "hi": 6.0},

    # How the two jumps are sampled.
    #   "fractional": delta_eps_trans/eps_trans  and  delta_eps_1/eps_1 uniform in
    #                 FRAC_* ranges below (DEFAULT, agreed).
    #   "absolute"  : delta_eps uniform in ABS_* ranges [MeV/fm^3].
    # NOTE: switching changes the prior volume and therefore the Bayes factors.
    "delta_mode": "fractional",
    "frac_delta_trans": (0.0, 1.5),
    "frac_delta_1": (0.0, 1.5),
    "abs_delta_trans": (0.0, 1000.0),
    "abs_delta_1": (0.0, 1000.0),

    # eps_1 is the top of polytrope 1 (pre-jump side), and must exceed
    # eps_trans + delta_eps_trans (the bottom of polytrope 1).  It is sampled
    # uniformly *conditional* on that.
    #   "width"   : eps_1 - start uniform in eps1_width, start = eps_trans+delta_eps_trans
    #               (never empty; DEFAULT)
    #   "absolute": eps_1 in [max(start+EPS1_MIN_GAP, EPS1_ABS[0]), EPS1_ABS[1]];
    #               points where this interval is empty get zero prior support
    "eps1_mode": "width",
    "eps1_width": (10.0, 1500.0),
    "eps1_abs": (300.0, 2500.0),
    "eps1_min_gap": 10.0,
    # Hard ceiling on eps_1, applied in BOTH modes.  The agreed specification put
    # eps_1 at "up to about 1800 MeV/fm^3"; without the cap the width prior sends
    # 54% of draws above it and puts the bottom of polytrope 2 beyond any central
    # density a stable star reaches, so those draws are a one-transition model
    # wearing two-transition clothes and they dilute the evidence.  Set to None to
    # recover the uncapped behaviour -- that changes the prior volume and therefore
    # every Bayes factor, so it is a decision to record, not a tuning knob.
    "eps1_cap": 1800.0,

    # polytropic indices of  P = K eps^gamma  (same wide range for both, so any
    # soft-then-stiff ordering is read off the posterior, not imposed)
    "gamma1": (0.5, 10.0),
    "gamma2": (0.5, 10.0),
}

# --------------------------------------------------------------------------
# 3. TOV / tidal sweep
# --------------------------------------------------------------------------
SWEEP = {
    "eps_c_min": 120.0,      # lowest central energy density in the sweep [MeV/fm^3]
    "eps_c_max": 3000.0,     # initial highest central energy density; extended
                             # automatically if M is still rising at the top
    "n_base": 70,            # log-spaced central pressures
    "cluster_deltas": (1e-3, 3e-3, 1e-2, 3e-2),  # extra points around each transition pressure
    "refine_rounds": 2,      # bisection refinement around every M(Pc) extremum
    "max_extensions": 4,
    "rtol": 1e-6,            # your solver used 1e-8; see validate.py for the convergence test
    "atol_M": 1e-10,
    "atol_P": 1e-10,
    "atol_y": 1e-8,
    "min_branch_dM": 0.01,   # a stable branch must span at least this much mass [Msun]
    # Resolution floor.  Two stars closer than min_dlnPc in ln Pc differ in mass by
    # less than the ODE error, so their dM sign is noise: the grid is thinned to this
    # spacing and |dM| <= noise_factor*rtol*M_max is treated as "flat" when branches
    # are identified.  Raising rtol without raising these two is unsafe.
    "min_dlnPc": 1e-3,
    "noise_factor": 50.0,
    "causality_tol": 1e-6,   # reject EOS if cs^2 > 1 + tol anywhere inside stable stars
}

# --------------------------------------------------------------------------
# 4. Observational data.
#
# Every entry carries its own column map, md5 and DOI.  THE COLUMN ORDER IS NOT
# THE SAME ACROSS RELEASES -- J0740 and J0030 are (weight, M, R) while the Miller
# J0437 file is (R, M, weight) -- and a transposition is silent, so each source
# also declares the published headline numbers, which datasets.check_headline
# re-derives from the loaded arrays at construction time.
#
# `bounds` are hard edges of the SAMPLING prior (not of physics): the KDE mirrors
# samples through them instead of leaking probability past them.
# --------------------------------------------------------------------------
_NICER = "../NICER PSR "
_LVK = "../LVK GW170817/"

DATA = {
    "mr_sources": {
        "J0740": {
            "use": True,
            "file": _NICER + "J0740+6620/J0740_gamma_NxX_lp40k_se001_wmrsamples copy.dat",
            "md5": "f3470465f6a9bddd8f3cf46fd6975240",
            "doi": "10.5281/zenodo.10519473  (Salmi et al. 2024, arXiv:2406.14466)",
            "cols": {"weight": 0, "M": 1, "R": 2},
            "order": ("M", "R"),
            "headline": {"M": (2.073, 0.069, 0.069), "R": (12.49, 0.88, 1.28)},
            "bounds": ((1, 16.0, "hi"),),      # R prior truncated at 16 km
            "note": "mass prior is the Fonseca+2021 radio/Shapiro measurement; that is "
                    "independent data and is deliberately NOT divided out.",
        },
        "J0030": {
            "use": True,
            "file": _NICER + "J0030+0451/weighted_samples_PDTU.txt",
            "md5": "e2d50fdca6e5a4a2ac447d2e6afe1d26",
            "doi": "Kini et al. 2026 (arXiv:2602.23743), Zenodo 10.5281/zenodo.18741941 -- PDT-U",
            "cols": {"weight": 0, "M": 1, "R": 2},
            "order": ("M", "R"),
            "headline": {"M": (1.434, 0.167, 0.204), "R": (12.68, 1.04, 1.30)},
            "bounds": ((0, 1.0, "lo"), (1, 16.0, "hi")),   # M and R prior edges
            "note": "uniform M prior on [1, 3]; the EOS-side mass prior must not "
                    "extend below 1.0 (config.LIKELIHOOD['mass_prior_min']).",
        },
        "J0437": {
            "use": True,
            "file": _NICER + "J0437\u20134715/J0437_NICER_RM.txt",
            "md5": "2e70f5f9a9bf35aaead616df581ee584",
            "doi": "Miller et al. 2026 (ApJL 1000 L48, arXiv:2512.08790), Zenodo 17833896",
            "cols": {"weight": 2, "M": 1, "R": 0},        # NOTE: (R, M, weight)
            "order": ("M", "R"),
            "headline": {"M": (1.429, 0.044, 0.044), "R": (14.12, 2.22, 1.36)},
            "bounds": (),
            "note": "mass prior is the Reardon+2024 radio measurement (kept, as above). "
                    "The radius posterior is bimodal -- see the Choudhury+2024 "
                    "co-primary scenario in the scenario list.",
        },
    },

    # GW170817.  Baseline: GWTC-1 IMRPhenomPv2NRT low spin.
    "gw": {
        "use": True,
        "file": _LVK + "GW170817_GWTC-1.hdf5",
        "md5": "58e919868f0fae590aad2348592e7ea2",
        "doi": "LIGO DCC P1800370, data DOI 10.7935/82H3-HH23 (GWTC-1)",
        "dataset": "IMRPhenomPv2NRT_lowSpin_posterior",
        # Source-frame masses.  Default: the host-galaxy redshift, which keeps the
        # source-frame chirp mass as sharp as the detector-frame one.  The
        # per-sample alternative inherits the GW distance uncertainty and is a
        # sensitivity test, not a correction.
        "redshift": {"mode": "fixed", "z": 0.0099, "H0": 67.90, "Om": 0.3065},
        "bounds": ((0, 1.0, "hi"), (1, 0.0, "lo"), (2, 0.0, "lo")),  # q<=1, Lambda>=0
        "note": "sampling prior verified from the release's own prior samples: "
                "Lambda1, Lambda2 flat on [0, 5000]; q NOT flat.",
    },

    # PSR J0952-0607 mass measurement (Romani et al. 2022)
    "mass_floor": {"mean": 2.35, "sigma": 0.17},
}

LIKELIHOOD = {
    # Joint prior over (mass, branch) when an EOS has several stable branches.
    #   "mixture" (default, approved) | "sum" | "max"   -- see likelihood.py
    "branch_rule": "mixture",
    # Lowest mass the population prior puts any weight on.  1.0 also matches the
    # lower edge of the J0030 sampling prior.
    "mass_prior_min": 1.0,
    # Quadrature points.  R(m) and Lambda(m) go like sqrt(m_max - m) at the end of
    # a branch, so the abscissae are clustered at the range ends, at each branch's
    # endpoints and over the data's support (likelihood.quad_grid).  With that,
    # 150 is converged to 3e-4 nat; with a uniform grid it was off by 0.08.
    "n_mass_grid": 150,
    "n_q_grid": 120,
    "q_min": 0.2,                # below this the GW posterior has no support
    # KDE bandwidths as a multiple of Silverman's rule at the EFFECTIVE sample size.
    # The NICER files have ESS 1e5-1e6 so they tolerate heavy under-smoothing;
    # GW170817 has 8078 equal-weight samples in 3D and does not.
    # validate.py test 8 reports the sensitivity of ln L to both.
    "bw_factor_mr": 0.3,
    "bw_factor_gw": 1.0,
    # EOS-side population prior on the binary mass ratio:
    #   "uniform_mass" (matches the GW sampling prior; the q-shapes then cancel)
    #   "flat_q"       (divides pi_s(q) out explicitly)
    "gw_population": "uniform_mass",
    # "cdf"      : P(m < M_TOV), the conventional one-sided floor (approved spec)
    # "marginal" : the same integral against pi(m|theta) as the NICER terms
    "mass_floor_mode": "cdf",
    "seed": 12345,
}

# --------------------------------------------------------------------------
# 5. Sampler
# --------------------------------------------------------------------------
SAMPLER = {
    "nlive": 500,
    "dlogz": 0.5,
    "sample": "auto",
    "bound": "multi",
    "nprocs": 1,
    "seed": 1,
    "outdir": "results",
}
