"""
analysis.py -- Stage G (posterior characterization) + Stage H (Bayes factors)
+ the HESS J1731-347 / XTE J1814-338 stress-test diagnostic (used as a
post-hoc boundary check, NOT folded into the likelihood -- per your
resolved decision).
"""

import pickle
import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as plt

from priors import PARAM_NAMES, PARAM_LABELS


def load_results(paths):
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            r = pickle.load(f)
        out[f"{r['label']}_{r['nicer_scenario']}"] = r
    return out


def credible_intervals(result):
    """Median + 68%/95% credible intervals per parameter, weighted by the
    nested-sampling importance weights (NOT a naive unweighted percentile --
    dynesty's samples carry unequal weight, so this has to use
    dynesty.utils.resample_equal or an explicit weighted quantile)."""
    from dynesty.utils import resample_equal
    weights = np.exp(result["logwt"] - result["logz"][-1])
    weights = weights / weights.sum()
    samples_eq = resample_equal(result["samples"], weights)

    rows = []
    for i, name in enumerate(PARAM_NAMES):
        col = samples_eq[:, i]
        q025, q16, q50, q84, q975 = np.percentile(col, [2.5, 16, 50, 84, 97.5])
        rows.append({
            "parameter": name, "median": q50,
            "68%_low": q16, "68%_high": q84,
            "95%_low": q025, "95%_high": q975,
        })
    return pd.DataFrame(rows), samples_eq


def correlation_matrix(samples_eq):
    """Pearson correlation across the equally-weighted posterior samples --
    this is where the masquerade-problem degeneracy (e.g. n_t1 vs. Gamma1
    trading off) shows up quantitatively, not just as a caveat."""
    df = pd.DataFrame(samples_eq, columns=PARAM_NAMES)
    return df.corr()


def make_corner_plot(result, savepath):
    from dynesty.utils import resample_equal
    weights = np.exp(result["logwt"] - result["logz"][-1])
    weights = weights / weights.sum()
    samples_eq = resample_equal(result["samples"], weights)
    fig = corner.corner(samples_eq, labels=PARAM_LABELS, show_titles=True,
                         title_fmt=".3g", quantiles=[0.16, 0.5, 0.84])
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return savepath


JEFFREYS_SCALE = [
    (0, 1.0, "inconclusive"),
    (1.0, 2.5, "weak evidence for the hybrid model"),
    (2.5, 5.0, "moderate evidence for the hybrid model"),
    (5.0, np.inf, "strong evidence for the hybrid model"),
]


def _jeffreys_label(ln_K):
    if ln_K < 0:
        return "favors the pure hadronic model"
    for lo, hi, label in JEFFREYS_SCALE:
        if lo <= ln_K < hi:
            return label
    return "strong evidence for the hybrid model"


def bayes_factor_table(all_results):
    rows = []
    for key, r in all_results.items():
        ln_K = r["log_bayes_factor"]
        rows.append({
            "family": r["label"], "NICER_scenario": r["nicer_scenario"],
            "logZ_hadronic": r["logZ_had"],
            "logZ_hybrid": r["logZ_hybrid"], "logZ_hybrid_err": r["logZ_hybrid_err"],
            "ln_K": ln_K, "interpretation": _jeffreys_label(ln_K),
        })
    return pd.DataFrame(rows).sort_values("ln_K", ascending=False)


# ---------------------------------------------------------------------------
# HESS J1731-347 / XTE J1814-338 -- post-hoc stress test, not a likelihood term
# ---------------------------------------------------------------------------
STRESS_TEST_POINTS = {
    "HESS J1731-347": {"M": 0.77, "M_lo": 0.60, "M_hi": 0.97, "R": 10.4, "R_lo": 9.62, "R_hi": 11.26},
    "XTE J1814-338": {"M": 1.21, "M_lo": 1.16, "M_hi": 1.26, "R": 7.0, "R_lo": 6.6, "R_hi": 7.4},
}


def stress_test_reach(hadronic, samples_eq, n_check=200, seed=0):
    """For a random subset of posterior hybrid-EOS draws, check whether the
    EOS's stable branch passes anywhere near the HESS J1731-347 / XTE
    J1814-338 boundary points -- reported as a separate diagnostic table,
    never as a likelihood contribution (per your resolved decision: these
    are too systematics-prone, and R^5-sensitive tidal fits would distort
    the canonical-mass fit if forced)."""
    import hybrid_eos
    import solver_interface

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples_eq), size=min(n_check, len(samples_eq)), replace=False)

    results = {name: {"reached": 0, "checked": 0, "min_R_at_M": []} for name in STRESS_TEST_POINTS}
    for i in idx:
        theta = dict(zip(PARAM_NAMES, samples_eq[i]))
        table = hybrid_eos.build_hybrid_eos(hadronic, theta)
        if not table["valid"]:
            continue
        curve = solver_interface.build_curve(table["P"], table["eps"])
        if not curve["ok"]:
            continue
        for name, pt in STRESS_TEST_POINTS.items():
            results[name]["checked"] += 1
            lo, hi = curve["M_range"]
            if lo <= pt["M"] <= hi:
                R_pred = float(curve["R_of_M"](pt["M"]))
                results[name]["min_R_at_M"].append(R_pred)
                if pt["R_lo"] <= R_pred <= pt["R_hi"]:
                    results[name]["reached"] += 1

    rows = []
    for name, r in results.items():
        rows.append({
            "source": name,
            "fraction_of_posterior_reaching_mass": r["checked"] and r["reached"] / max(r["checked"], 1),
            "n_checked": r["checked"],
            "median_predicted_R_at_M": float(np.median(r["min_R_at_M"])) if r["min_R_at_M"] else np.nan,
            "observed_R_range": (STRESS_TEST_POINTS[name]["R_lo"], STRESS_TEST_POINTS[name]["R_hi"]),
        })
    return pd.DataFrame(rows)
