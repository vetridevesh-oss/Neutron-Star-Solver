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
# 2. PRIOR RANGES  (uniform priors; the three sampled parameters are
#    eps_trans, delta_eps_trans, gamma)
#
# eps_trans : energy density at the top of the hadronic phase (pre-jump side)
#     kind = "n0"      -> bounds given in units of n0, converted to energy
#                         density with the hadronic table's own n(eps) column
#     kind = "MeV/fm3" -> bounds are energy densities directly
# --------------------------------------------------------------------------
PRIOR = {
    "eps_trans": {"kind": "n0", "lo": 1.0, "hi": 6.0},

    # How the jump is sampled.
    #   "fractional": delta_eps_trans/eps_trans uniform in frac_delta_trans (DEFAULT, agreed).
    #   "absolute"  : delta_eps_trans uniform in abs_delta_trans [MeV/fm^3].
    # NOTE: switching changes the prior volume and therefore the Bayes factors.
    "delta_mode": "fractional",
    "frac_delta_trans": (0.0, 1.5),
    "abs_delta_trans": (0.0, 1000.0),

    # polytropic index of  P = K eps^gamma  above the transition.  The polytrope
    # has no upper end, so gamma alone sets how stiff the core is all the way to
    # the central density of the heaviest stable star -- which is why most of the
    # prior that gets rejected is rejected for cs^2 > 1 inside a star.
    "gamma": (0.5, 10.0),
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
# FROZEN 2026-11-01 (see PREREGISTRATION.md).  The release chosen for each
# source is a pre-registered decision, not a default: J0030 = Kini+2026 PDT-U,
# J0437 = Miller+2026, GW170817 = GWTC-1 IMRPhenomPv2NRT low-spin.  Swapping one
# in produces a SENSITIVITY number, which must be labelled as such.
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
            # X-PSI sampling prior: the Fonseca+2021 radio Gaussian in M times a
            # flat prior in R_eq.  Flat in R is a constant, and the Gaussian in M
            # is independent data we keep, so there is nothing to divide out.
            "sampling_prior": "radio Gaussian in M x flat in R_eq; nothing divided out",
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
            # Joint flat in (M, R_eq) with hard edges (the Vinciguerra+2024
            # convention, assumed to carry over to Kini+2026 -- CHECK against the
            # Zenodo prior samples).  A constant, so nothing is divided out; the
            # edges are handled by the KDE's mirroring (`bounds` above).
            "sampling_prior": "joint flat in (M, R_eq) with hard edges; constant, nothing divided out",
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
            # Miller+2026 Table 1: flat in x = c^2 R_eq/(GM) on [3.2, 8.0] times the
            # Reardon+2024 radio Gaussian N(1.418, 0.044) in M.  Dividing the flat-in-x
            # part out would reweight by a factor proportional to M, which moves the
            # weight by only +/-3% across this posterior, so it is NOT applied.  The
            # x = 8 edge is the diagonal R = 11.81 (M/Msun) km and is not mirrored by
            # `bounds`; 1.6% of the weight lies within 0.25 km of it, against a kernel
            # width of about 0.06 km, so the leakage is negligible.
            "sampling_prior": "flat in c^2 R/(GM) on [3.2, 8] x radio Gaussian in M; "
                              "the implied weight ~ M (<=3%) is omitted",
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

    # PSR J0952-0607 mass.  Romani et al. 2025 (arXiv:2512.05099) tighten the
    # Romani et al. 2022 value 2.35 +/- 0.17 to 2.35 +/- 0.11 (optical light-curve
    # modelling of the companion; the quoted sigma does not include systematics).
    # 0.17 is kept as a sensitivity scenario.
    "mass_floor": {"mean": 2.35, "sigma": 0.11,
                   "doi": "arXiv:2512.05099 (Romani+2025); 0.17 from arXiv:2207.05124"},
}

LIKELIHOOD = {
    # Joint prior over (mass, branch) when an EOS has several stable branches.
    #   "mixture" | "sum" | "max"   -- see likelihood.py
    # FROZEN 2026-11-01 as "mixture": only 2.2% of accepted draws are multi-branch
    # and on those all three rules agree to 0.01 nat, so the rule does no
    # measurable work; "mixture" is the one that is a proper marginal.
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
    # "cdf"      : P(m < M_TOV), the conventional one-sided floor
    # "marginal" : the same integral against pi(m|theta) as the NICER terms
    # DEFAULT is "marginal" (decided 2026-10-31): every pulsar term then uses the
    # same EOS-limited population prior, flat in m on the stable range normalised
    # by 1/|U| (Legred+2021, arXiv:2106.05313).  "cdf" corresponds to an improper
    # flat prior on m and carries no Occam factor; it is kept only to reproduce the
    # older numbers.
    "mass_floor_mode": "marginal",
    "seed": 12345,
}

# --------------------------------------------------------------------------
# 4b. Prior acceptance (evidence renormalisation).
#
# The three causality screens truncate the prior: those draws return
# ln L = -inf, so dynesty's evidence is an integral over the WHOLE unit
# cube and comes out as  ln Z_cube = ln f_acc + ln Z_truncated.  The truncated
# prior is the one the model means, so run_inference subtracts ln f_acc, which
# acceptance.py estimates per hadronic table by Monte Carlo through the same
# prior transform and the same screens.
# --------------------------------------------------------------------------
ACCEPTANCE = {"n_draws": 4000, "seed": 7}

# --------------------------------------------------------------------------
# 5. Sampler
# --------------------------------------------------------------------------
SAMPLER = {
    "nlive": 500,
    # 0.1, not 0.5: dlogz bounds the evidence still sitting in the live points,
    # and the ln B differences this study reports are order 1 nat, so a 0.5
    # stopping criterion is the same size as the signal.  FROZEN 2026-11-01.
    "dlogz": 0.1,
    "sample": "auto",
    "bound": "multi",
    "nprocs": 1,
    "seed": 1,
    "outdir": "results",
}
