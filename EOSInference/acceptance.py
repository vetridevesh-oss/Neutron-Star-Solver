"""
acceptance.py -- the fraction of the hybrid prior that survives the hard constraints.

The prior is truncated: draws that violate any of the three causality screens
(or that are not a valid EOS at all) get ln L = -inf (see
model.HybridModel.screen).  dynesty
integrates the likelihood over the WHOLE unit cube, so what it returns is

    ln Z_cube = ln f_acc + ln Z_truncated,

while the model means the truncated prior -- the one that is normalised over the
EOS families the spec actually allows.  run_inference therefore subtracts
ln f_acc.  Both terms are needed for the Bayes factor: the hadronic baseline has
no free parameters and no truncation, so leaving f_acc in would hand the hybrid
model a penalty of order one nat that has nothing to do with the data.

f_acc depends on the hadronic table (P_t, the hadronic causality bound and the
reachable central densities all come from it), so it is estimated per table by
Monte Carlo through the SAME prior transform and the SAME screens as the sampler
uses, and cached next to that table's results.

Solver failures are a numerical outcome, not a statement about the prior, so they
are dropped from the denominator and reported separately.

Measured for the one-transition model, 2026-10-31 (n = 3000, seed 7):
    FSU2L80  f_acc = 0.335 +/- 0.009   (-ln f_acc = 1.09 nat)
    NL3      f_acc = 0.216 +/- 0.008   (-ln f_acc = 1.53 nat)
Most of what is rejected is rejected for cs^2 > 1: the single polytrope runs
all the way to the centre of the heaviest star, so a stiff gamma is acausal
there even when it is causal at the transition.
"""
import argparse
import collections
import json
import os

import numpy as np

import config
from model import PRIOR_REJECTIONS, SCREEN_KINDS, HybridModel


def classify(model, u):
    """Classify one unit-cube point.  Returns (kind, theta).

    The classification is model.screen() itself, so it cannot drift from the
    filters dynesty sees through model.loglike.
    """
    theta = model.prior_transform(np.asarray(u, dtype=float))
    return model.screen(theta)[0], theta


class _Classifier:
    """Picklable worker that rebuilds a likelihood-free model once per process.

    Classification needs the hadronic table, the prior transform and the solver,
    but never the likelihood, so what crosses the process boundary is three small
    config objects instead of the KDEs.  The model is cached per process, so the
    table is read and numba's cache is warmed once, not once per draw.

    The sweep and prior settings travel with the worker, but the two table-reading
    constants (config.EOS_SKIPROWS, config.EOS_COLS) are re-read from config in the
    child, so overriding THOSE in a notebook would not reach a spawned worker.
    validate.py test 10 compares the pooled and serial classifications.
    """

    _cache = {}

    def __init__(self, eos_file, sweep_cfg, prior_cfg):
        self.eos_file, self.sweep_cfg, self.prior_cfg = eos_file, sweep_cfg, prior_cfg

    @property
    def model(self):
        key = (self.eos_file, repr(self.sweep_cfg), repr(self.prior_cfg))
        m = _Classifier._cache.get(key)
        if m is None:
            m = HybridModel(self.eos_file, None, sweep_cfg=self.sweep_cfg,
                            prior=self.prior_cfg)
            _Classifier._cache[key] = m
        return m

    def __call__(self, u):
        kind, theta = classify(self.model, u)
        return kind, np.asarray(theta, dtype=float)


def _cache_matches(d, eos_file, n_draws, seed):
    return (d.get("eos") == eos_file and d.get("n_draws") == n_draws
            and d.get("seed") == seed)


def estimate(model, n_draws=None, seed=None, cache=None, pool=None, verbose=False):
    """Monte-Carlo estimate of the surviving prior fraction for `model`'s table.

    Draws u ~ U(0,1)^ndim, pushes them through model.prior_transform and classifies
    them with model.screen.  Returns a JSON-able dict with f_acc, its binomial
    error, the rejection breakdown and P(gamma1 < gamma2) among accepted draws.

    `cache` is a path; a cached result is reused only if it was computed for the
    same table with the same n_draws and seed.  `pool` is anything with a .map
    (dynesty.pool.Pool included).
    """
    n_draws = config.ACCEPTANCE["n_draws"] if n_draws is None else int(n_draws)
    seed = config.ACCEPTANCE["seed"] if seed is None else int(seed)
    if cache and os.path.exists(cache):
        with open(cache) as f:
            old = json.load(f)
        if _cache_matches(old, model.eos_file, n_draws, seed):
            return old

    U = np.random.default_rng(seed).random((n_draws, model.ndim))
    if pool is None:
        res = [classify(model, u) for u in U]
    else:
        worker = _Classifier(model.eos_file, model.sweep_cfg, model.prior_transform.p)
        res = pool.map(worker, list(U))

    kinds = [r[0] for r in res]
    counts = collections.Counter(kinds)
    unknown = set(counts) - set(SCREEN_KINDS)
    if unknown:
        raise RuntimeError("model.screen returned unknown kinds: %s" % sorted(unknown))
    n_fail = counts.get("solver_failure", 0)
    n = n_draws - n_fail                       # solver failures leave the denominator
    n_acc = counts.get("accepted", 0)
    if n <= 0:
        raise RuntimeError("every draw failed in the solver; nothing to estimate from")
    f = n_acc / n
    out = dict(eos=model.eos_file, f_acc=float(f),
               f_acc_err=float(np.sqrt(f * (1 - f) / n)),
               ln_f_acc=float(np.log(f)) if f > 0 else float("-inf"),
               n_draws=n_draws, n_used=int(n), n_accepted=int(n_acc),
               n_solver_failures=int(n_fail), seed=seed,
               counts={k: int(counts.get(k, 0)) for k in SCREEN_KINDS},
               rejection_fractions={k: counts.get(k, 0) / n for k in PRIOR_REJECTIONS})
    if verbose:
        print(describe(out))
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        with open(cache, "w") as fh:
            json.dump(out, fh, indent=2)
    return out


def describe(acc):
    """One-paragraph summary of an estimate() result."""
    rej = "  ".join("%s %.1f%%" % (k.replace("acausal_", "ac:"), 100 * v)
                    for k, v in sorted(acc["rejection_fractions"].items(),
                                       key=lambda kv: -kv[1]))
    return ("prior acceptance for {eos}: f_acc = {f_acc:.3f} +/- {f_acc_err:.3f} "
            "(-ln f_acc = {mlnf:.2f} nat) from {n_used} of {n_draws} draws, seed {seed}"
            "\n   rejections: {rej}"
            "\n   solver failures (excluded): {n_solver_failures}").format(
        mlnf=-acc["ln_f_acc"], rej=rej, **acc)


def _main():
    ap = argparse.ArgumentParser(description="measure the surviving prior fraction")
    ap.add_argument("--eos", nargs="*", default=None,
                    help="hadronic EOS file(s); default config.HADRONIC_EOS_FILES")
    ap.add_argument("--n-draws", type=int, default=config.ACCEPTANCE["n_draws"])
    ap.add_argument("--seed", type=int, default=config.ACCEPTANCE["seed"])
    ap.add_argument("--nprocs", type=int, default=1)
    ap.add_argument("--cache", default=None, help="write/reuse this JSON file")
    args = ap.parse_args()
    for eos in (args.eos or config.HADRONIC_EOS_FILES):
        model = HybridModel(eos, None)
        if args.nprocs > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(args.nprocs) as pool:
                acc = estimate(model, args.n_draws, args.seed, cache=args.cache,
                               pool=pool, verbose=True)
        else:
            acc = estimate(model, args.n_draws, args.seed, cache=args.cache, verbose=True)
        print(json.dumps({k: acc[k] for k in ("f_acc", "f_acc_err", "ln_f_acc")}, indent=None))


if __name__ == "__main__":
    _main()
