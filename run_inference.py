"""
run_inference.py -- Stage F. For ONE hadronic base EOS (fixed L), get:
    Z_had    = hadronic_null_loglike(...)        (single evaluation, no sampling)
    Z_hybrid = dynesty evidence over the 6-parameter crossover model

Run this once per held-out hadronic family (FSU2L, IUFSU, FSUGarnet, RMF022,
TAMUa/b/c, NL3, FSUGold2, RMF032 -- or here, GM1/TM1/NL3wr4/DD2 per the
mentee's spec) and collect the (Z_had, Z_hybrid) pairs for Stage H's Bayes
factor table.

PARALLELIZATION: dynesty's own `pool=` parallelizes ACROSS live points
(i.e. across likelihood calls), with each individual call running the
in-memory forward solve serially inside build_curve(). This is the split
discussed earlier in the project -- do NOT nest a ProcessPoolExecutor
inside log_likelihood itself; daemonic pool workers can't spawn children.
"""

import pickle
import numpy as np
import dynesty

from hybrid_eos import HadronicBaseline
from priors import make_prior_transform, PARAM_NAMES
from likelihood import make_log_likelihood, hadronic_null_loglike


def run_one_family(hadronic_path, sources, label, nicer_scenario="C",
                    nlive=250, pool_size=None, dlogz=0.5, save_dir="results"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    hadronic = HadronicBaseline(hadronic_path)
    prior_transform = make_prior_transform(hadronic)
    log_likelihood = make_log_likelihood(hadronic, sources, nicer_scenario=nicer_scenario)

    logZ_had = hadronic_null_loglike(hadronic, sources, nicer_scenario=nicer_scenario)

    sampler_kwargs = dict(ndim=6, nlive=nlive)
    if pool_size and pool_size > 1:
        # dynesty needs plain top-level functions to pickle to pool workers;
        # log_likelihood/prior_transform above are closures, which is fine
        # for dynesty.pool.Pool IF the whole module is importable (not
        # defined inside __main__) -- keep this driver in its own script.
        with dynesty.pool.Pool(pool_size, log_likelihood, prior_transform) as pool:
            sampler = dynesty.NestedSampler(
                pool.loglike, pool.prior_transform, pool=pool, **sampler_kwargs)
            sampler.run_nested(dlogz=dlogz)
    else:
        sampler = dynesty.NestedSampler(log_likelihood, prior_transform, **sampler_kwargs)
        sampler.run_nested(dlogz=dlogz)

    results = sampler.results
    logZ_hybrid = float(results.logz[-1])
    logZ_hybrid_err = float(results.logzerr[-1])

    out = {
        "label": label, "nicer_scenario": nicer_scenario,
        "logZ_had": logZ_had, "logZ_hybrid": logZ_hybrid, "logZ_hybrid_err": logZ_hybrid_err,
        "log_bayes_factor": logZ_hybrid - logZ_had,
        "samples": results.samples, "logwt": results.logwt, "logz": results.logz,
        "param_names": PARAM_NAMES,
    }
    with open(f"{save_dir}/{label}_{nicer_scenario}.pkl", "wb") as f:
        pickle.dump(out, f)
    return out


if __name__ == "__main__":
    print(
        "Usage pattern:\n"
        "    from fetch_observational_data import make_placeholder_sources  # or real, loaded sources\n"
        "    sources = make_placeholder_sources()\n"
        "    run_one_family('GM1.dat', sources, label='GM1', nicer_scenario='C')\n"
        "    run_one_family('GM1.dat', sources, label='GM1', nicer_scenario='B')  # ablation\n"
        "Repeat per hadronic family, then hand the saved results/*.pkl files to analysis.py."
    )
