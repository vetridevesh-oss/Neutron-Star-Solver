"""
priors.py -- unit-cube -> physical-parameter transform for the 6 free
crossover parameters, for use as dynesty's `prior_transform`.

All 6 priors are uniform, per the mentee's spec, EXCEPT that d_eps_core's
range depends on eps_trans = eps_Hadronic(P_trans), which depends on the
just-sampled P_trans and d_eps -- so this is a CONDITIONAL prior, not an
independent one. dynesty (and any nested sampler) is fine with this as long
as the transform is applied in dependency order within a single call, which
is what happens below: P_trans and d_eps_frac are resolved first, eps_trans
is looked up from the hadronic table, THEN d_eps_core's uniform range is
computed and applied to its own unit-cube coordinate.

Parameter order (must match the order used everywhere else -- likelihood,
corner plot labels, credible-interval tables):
    0: P_trans        [MeV/fm^3]
    1: delta_p_frac   dimensionless (delta_p / P_trans)
    2: d_eps_frac     dimensionless (Delta_eps / eps_trans)
    3: gamma2         dimensionless
    4: d_eps_core     [MeV/fm^3]
    5: gamma3         dimensionless
"""

import numpy as np
from hybrid_eos import EPS_NS, EPS_MAX

PARAM_NAMES = ["P_trans", "delta_p_frac", "d_eps_frac", "gamma2", "d_eps_core", "gamma3"]
PARAM_LABELS = [r"$P_{\rm trans}$ [MeV/fm$^3$]", r"$\delta p / P_{\rm trans}$",
                r"$\Delta\varepsilon/\varepsilon_{\rm trans}$", r"$\gamma_2$",
                r"$\Delta\varepsilon_{\rm core}$ [MeV/fm$^3$]", r"$\gamma_3$"]

P_TRANS_RANGE = (10.0, 400.0)      # MeV/fm^3
DELTA_P_FRAC_RANGE = (0.01, 0.25)
D_EPS_FRAC_RANGE = (0.0, 3.0)
GAMMA2_RANGE = (1.0, 8.0)
GAMMA3_RANGE = (0.5, 8.0)


def make_prior_transform(hadronic):
    """hadronic: a hybrid_eos.HadronicBaseline instance, needed to evaluate
    eps_trans(P_trans) inside the transform."""

    def prior_transform(u):
        theta = np.empty(6)

        theta[0] = P_TRANS_RANGE[0] + u[0] * (P_TRANS_RANGE[1] - P_TRANS_RANGE[0])
        P_trans = theta[0]

        theta[1] = DELTA_P_FRAC_RANGE[0] + u[1] * (DELTA_P_FRAC_RANGE[1] - DELTA_P_FRAC_RANGE[0])

        eps_trans = float(hadronic.eps(np.array([P_trans]))[0])

        theta[2] = D_EPS_FRAC_RANGE[0] + u[2] * (D_EPS_FRAC_RANGE[1] - D_EPS_FRAC_RANGE[0])
        d_eps = theta[2] * eps_trans

        theta[3] = GAMMA2_RANGE[0] + u[3] * (GAMMA2_RANGE[1] - GAMMA2_RANGE[0])

        # Conditional range for d_eps_core, per the spec:
        #   U(0, eps_max - (eps_trans + d_eps))
        # Clipped at a small positive floor if a pathological (P_trans,
        # d_eps) combination would otherwise push eps_trans+d_eps past
        # eps_max -- this is a genuine prior-support edge case (see
        # hybrid_eos.py module docstring assumption discussion); flagged
        # here rather than silently producing a negative range.
        upper = max(EPS_MAX - (eps_trans + d_eps), 1e-3)
        theta[4] = u[4] * upper

        theta[5] = GAMMA3_RANGE[0] + u[5] * (GAMMA3_RANGE[1] - GAMMA3_RANGE[0])

        return theta

    return prior_transform


def theta_array_to_dict(theta):
    return dict(zip(PARAM_NAMES, theta))
