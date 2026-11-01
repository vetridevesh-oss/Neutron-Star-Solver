"""
run_inference.py -- nested-sampling Bayesian inference of the hybrid core.

    python run_inference.py                       # all EOS files in config.HADRONIC_EOS_FILES
    python run_inference.py --eos EOSBetaFSU2L80.dat --nlive 800 --nprocs 8
    python run_inference.py --rule max            # branch rule: mixture | max | sum
    python run_inference.py --mock                # synthetic data (pipeline test, no real data needed)
    python run_inference.py --resume              # continue from the checkpoint

For each hadronic EOS it runs dynesty over the three parameters
(eps_trans, delta_eps_trans, gamma), evaluates the
pure-hadronic baseline (no free parameters => Z_had = L_had) and reports
    ln B(hybrid / hadronic) = ln Z_hybrid - ln L_had.
ln Z_hybrid is the evidence under the TRUNCATED prior: dynesty integrates over the
whole unit cube and the rejected draws contribute zero, so its own ln Z is
ln f_acc + ln Z_truncated and ln f_acc is subtracted here (see acceptance.py).
Output (in <outdir>/<eos-name>/): samples.npz, summary.json, acceptance.json
(+ dynesty checkpoint).
"""
import argparse
import json
import os
import time

import numpy as np

import acceptance
import config
from likelihood import Likelihood
from model import HybridModel
from priors import PARAM_NAMES


def make_likelihood(args):
    if args.mock:
        import solver, mockdata
        from hybrid_eos import HadronicTable, HybridEOS
        tab = HadronicTable(args.eos_list[0], config.EOS_SKIPROWS, config.EOS_COLS)
        seq = solver.compute_sequence(HybridEOS.pure_hadronic(tab))
        print("[mock] synthetic data drawn around the pure hadronic sequence of", args.eos_list[0])
        like = mockdata.make_mock_likelihood(seq, seed=0, like_cfg=config.LIKELIHOOD)
    else:
        like = Likelihood(like_cfg=config.LIKELIHOOD, verbose=True)
    print(like.describe())
    return like


def run_one(eos_file, like, args):
    import dynesty
    sm = config.SAMPLER
    name = os.path.splitext(os.path.basename(eos_file))[0]
    outdir = os.path.join(args.outdir, name)
    os.makedirs(outdir, exist_ok=True)
    model = HybridModel(eos_file, like)

    lnL_had, seq_had = model.hadronic_baseline()
    print(f"[{name}] pure hadronic: ln L = {lnL_had:.3f}  (M_TOV = {seq_had.M_tov:.3f})")
    if not np.isfinite(lnL_had):
        print(f"[{name}] WARNING: the hadronic baseline itself is rejected ({seq_had.reason}); "
              "ln B is undefined for this EOS.")

    rng = np.random.default_rng(args.seed)
    ckpt = os.path.join(outdir, "dynesty.save")
    acc_cache = os.path.join(outdir, "acceptance.json")
    kw = dict(nlive=args.nlive, sample=args.sample, bound=args.bound, rstate=rng)
    t0 = time.time()
    if args.nprocs > 1:
        import dynesty.pool as dypool
        with dypool.Pool(args.nprocs, model.loglike, model.prior_transform) as pool:
            sampler = _make(dynesty, args, pool.loglike, pool.prior_transform, kw, pool, ckpt)
            _go(sampler, args, ckpt)
            wall = time.time() - t0
            res = sampler.results
            # thousands of independent TOV solves: reuse the sampler's pool while
            # it is still alive (the workers rebuild a likelihood-free model)
            acc = acceptance.estimate(model, cache=acc_cache, pool=pool, verbose=True)
    else:
        sampler = _make(dynesty, args, model.loglike, model.prior_transform, kw, None, ckpt)
        _go(sampler, args, ckpt)
        wall = time.time() - t0
        res = sampler.results
        acc = acceptance.estimate(model, cache=acc_cache, verbose=True)

    # dynesty's ln Z is an integral over the whole unit cube with L = 0 on every
    # draw the screens reject, i.e. ln Z_cube = ln f_acc + ln Z_truncated.  The
    # model is defined on the truncated prior -- and the hadronic baseline has no
    # truncation at all -- so the Bayes factor uses the corrected value.
    logz_cube, logzerr_cube = float(res.logz[-1]), float(res.logzerr[-1])
    ln_f, ln_f_err = acc["ln_f_acc"], acc["f_acc_err"] / acc["f_acc"]
    logz = logz_cube - ln_f
    logzerr = float(np.hypot(logzerr_cube, ln_f_err))
    weights = np.exp(res.logwt - res.logz[-1])
    summary = dict(eos=eos_file, branch_rule=config.LIKELIHOOD["branch_rule"], nlive=args.nlive,
                   ncall=int(np.sum(res.ncall)), wall_s=wall, logZ=logz, logZ_err=logzerr,
                   logZ_unit_cube=logz_cube, logZ_unit_cube_err=logzerr_cube,
                   prior_acceptance=_jsonable(acc),
                   lnL_hadronic=float(lnL_had), lnB_hybrid_vs_hadronic=float(logz - lnL_had),
                   lnB_err=logzerr,
                   hadronic_components=_jsonable(like.components(seq_had)) if seq_had.ok else {},
                   data=_jsonable(getattr(like, "provenance", [])),
                   sweep=_jsonable(config.SWEEP),
                   prior=_jsonable(config.PRIOR), likelihood=_jsonable(config.LIKELIHOOD))
    np.savez(os.path.join(outdir, "samples.npz"), samples=res.samples, logl=res.logl, weights=weights,
             logz=logz, names=np.array(PARAM_NAMES))
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{name}] ln Z = {logz:.3f} +/- {logzerr:.3f} "
          f"(unit cube {logz_cube:.3f}, prior acceptance {acc['f_acc']:.3f} -> {-ln_f:+.3f}); "
          f"ln B(hybrid/hadronic) = {logz - lnL_had:+.3f}; "
          f"{summary['ncall']} calls in {wall/60:.1f} min")
    return summary


def _make(dynesty, args, loglike, ptform, kw, pool, ckpt):
    if args.resume and os.path.exists(ckpt):
        print("resuming from", ckpt)
        return dynesty.NestedSampler.restore(ckpt, pool=pool)
    extra = dict(pool=pool, queue_size=args.nprocs) if pool is not None else {}
    return dynesty.NestedSampler(loglike, ptform, len(PARAM_NAMES), **kw, **extra)


def _go(sampler, args, ckpt):
    sampler.run_nested(dlogz=args.dlogz, checkpoint_file=ckpt, print_progress=True,
                       **({"maxcall": args.maxcall} if args.maxcall else {}))


def _jsonable(d):
    return json.loads(json.dumps(d, default=str))


def main():
    sm = config.SAMPLER
    ap = argparse.ArgumentParser()
    ap.add_argument("--eos", nargs="*", help="hadronic EOS file(s); default config.HADRONIC_EOS_FILES")
    ap.add_argument("--nlive", type=int, default=sm["nlive"])
    ap.add_argument("--dlogz", type=float, default=sm["dlogz"])
    ap.add_argument("--sample", default=sm["sample"])
    ap.add_argument("--bound", default=sm["bound"])
    ap.add_argument("--nprocs", type=int, default=sm["nprocs"])
    ap.add_argument("--seed", type=int, default=sm["seed"])
    ap.add_argument("--outdir", default=sm["outdir"])
    ap.add_argument("--rule", choices=["mixture", "max", "sum"], help="override config branch rule")
    ap.add_argument("--maxcall", type=int, default=0, help="stop after this many likelihood calls (tests)")
    ap.add_argument("--mock", action="store_true", help="use synthetic data (pipeline test)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.eos_list = args.eos or config.HADRONIC_EOS_FILES
    if args.rule:
        config.LIKELIHOOD["branch_rule"] = args.rule
    like = make_likelihood(args)
    for f in args.eos_list:
        run_one(f, like, args)


if __name__ == "__main__":
    main()
