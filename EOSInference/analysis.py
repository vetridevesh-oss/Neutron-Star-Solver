"""
analysis.py -- post-processing of a run_inference.py result.

    python analysis.py results/EOSBetaFSU2L80 [--ndraw 300]

Prints the weighted posterior summary, ln B against the pure hadronic EOS, and
the diagnostics that decide whether that ln B means anything:

  * how often each transition is actually *probed* -- i.e. whether any stable
    star in the posterior draw reaches P_t and P_1.  A draw whose second
    transition sits above every stable central density is a one-transition
    model in disguise and tells you nothing about two transitions.
  * the number of stable branches, and the twin/third-family fraction.
  * where the posterior sits relative to the Seidov line, which controls how
    much twin volume the prior had access to.
  * the transition densities and chemical potentials in physical units
    (n/n0, mu in MeV), which is what a referee will ask for.

Writes corner.png and mr_band.png.
"""
import argparse
import collections
import json
import os

import numpy as np

import config
from hybrid_eos import HybridEOS
from model import HybridModel
from priors import PARAM_LABELS, PARAM_NAMES


def wquantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return np.interp(q, cw, x[o])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--ndraw", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = np.load(os.path.join(args.outdir, "samples.npz"))
    summ = json.load(open(os.path.join(args.outdir, "summary.json")))
    x, w = d["samples"], d["weights"]
    w = w / w.sum()

    print(f"EOS {summ['eos']} | branch rule: {summ['branch_rule']} | "
          f"ln Z = {summ['logZ']:.2f} +/- {summ['logZ_err']:.2f} | "
          f"ln B(hybrid/hadronic) = {summ['lnB_hybrid_vs_hadronic']:+.2f} "
          f"+/- {summ.get('lnB_err', summ['logZ_err']):.2f}")
    if summ.get("hadronic_components"):
        print("   hadronic baseline by term: " +
              "  ".join(f"{k}={v:+.2f}" for k, v in summ["hadronic_components"].items()))
    print(f"effective sample size: {1/np.sum(w**2):.0f}")
    print(f"{'param':>16} {'median':>9} {'68% low':>9} {'68% hi':>9} {'95% low':>9} {'95% hi':>9}")
    for i, n in enumerate(PARAM_NAMES):
        q = [wquantile(x[:, i], w, v) for v in (0.5, 0.16, 0.84, 0.025, 0.975)]
        print(f"{n:>16} {q[0]:9.2f} {q[1]:9.2f} {q[2]:9.2f} {q[3]:9.2f} {q[4]:9.2f}")
    for lab, d_, e_ in (("d_eps_trans/eps_trans", 1, 0), ("d_eps_1/eps_1", 3, 2)):
        r = x[:, d_] / x[:, e_]
        q = [wquantile(r, w, v) for v in (0.5, 0.16, 0.84)]
        print(f"{lab:>21} {q[0]:6.2f} (68%: {q[1]:.2f}-{q[2]:.2f})")

    try:
        import corner
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        corner.corner(x, weights=w, labels=PARAM_LABELS, quantiles=[0.16, 0.5, 0.84],
                      show_titles=True)
        plt.savefig(os.path.join(args.outdir, "corner.png"), dpi=120)
        plt.close()
    except Exception as e:
        print("corner plot skipped:", e)
        plt = None

    # ---- posterior-predictive diagnostics -------------------------------------
    import solver
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(x), args.ndraw, p=w)
    model = HybridModel(summ["eos"], None)
    topo = collections.Counter()
    probe1 = probe2 = hyb = n_ok = 0
    seidov1 = []
    Mtov, curves, nt, mu_t = [], [], [], []
    for i in idx:
        eos = HybridEOS(model.table, *x[i])
        if not eos.valid:
            continue
        seq = solver.compute_sequence(eos)
        if not seq.ok:
            continue
        n_ok += 1
        topo[seq.n_branches] += 1
        Mtov.append(seq.M_tov)
        Pc_max = seq.Pc_max_stable
        probe1 += Pc_max > eos.P_t          # some stable star contains polytrope-1 matter
        probe2 += Pc_max > eos.P_1          # ... and polytrope-2 matter
        hyb += seq.eps_c_tov > eos.params["eps_trans"]
        s = eos.seidov()[0]
        seidov1.append(s["ratio"] - s["threshold"])
        try:
            j = eos.number_density()["jumps"][0]
            nt.append(j["n_lo"] / config.N0)
            mu_t.append(j["mu"])
        except Exception:
            pass
        curves.append(seq)
    n = max(n_ok, 1)
    print(f"\nposterior-predictive diagnostics over {n_ok} draws")
    print("  stable branches per draw:", {k: f"{v/n:.1%}" for k, v in sorted(topo.items())})
    print(f"  first transition reached by some stable star : {probe1/n:.1%}")
    print(f"  second transition reached by some stable star: {probe2/n:.1%}"
          "   <- draws below this are effectively one-transition models")
    print(f"  M_TOV star has a hybrid core (eps_c > eps_trans): {hyb/n:.1%}")
    sd = np.array(seidov1)
    print(f"  Seidov margin at transition 1 (ratio - threshold): median {np.median(sd):+.3f}, "
          f"{100*np.mean(sd>0):.0f}% above the line (immediately unstable after the jump)")
    if nt:
        print(f"  transition 1: n_t = {np.median(nt):.2f} n0 "
              f"(68% {np.percentile(nt,16):.2f}-{np.percentile(nt,84):.2f}), "
              f"mu = {np.median(mu_t):.0f} MeV")
    print(f"  M_TOV: median {np.median(Mtov):.3f}, "
          f"68% {np.percentile(Mtov,16):.3f}-{np.percentile(Mtov,84):.3f}")

    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        for s in curves[:150]:
            for b in s.branches:
                ax.plot(b["R"], b["M"], color="C0", alpha=0.08, lw=1)
        ax.set_xlabel("R [km]")
        ax.set_ylabel(r"M [$M_\odot$]")
        ax.set_xlim(8, 16)
        ax.set_ylim(0.8, 3.0)
        plt.savefig(os.path.join(args.outdir, "mr_band.png"), dpi=120, bbox_inches="tight")
        print("wrote corner.png, mr_band.png")


if __name__ == "__main__":
    main()
