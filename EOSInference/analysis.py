"""
analysis.py -- post-processing of a run_inference.py result.

    python analysis.py results/EOSBetaFSU2L80 [--ndraw 300] [--nprior 300]

Prints the weighted posterior summary, ln B against the pure hadronic EOS, and
the diagnostics that decide whether that ln B means anything:

  * how often the transition is actually *probed* -- i.e. whether any stable
    star in the posterior draw reaches P_t at all.  A draw whose transition sits
    above every stable central density is a hadronic model in disguise and tells
    you nothing about a transition.
  * the number of stable branches, and the twin/third-family fraction.
  * where the posterior sits relative to the Seidov line, which controls how
    much twin volume the prior had access to.
  * the transition density and chemical potential in physical units
    (n/n0, mu in MeV), which is what a referee will ask for.

Writes five figures:

  corner.png       posterior corner, with the SCREENED PRIOR overlaid.  The
                   overlay is not decoration: three hard causality screens
                   reject about two thirds of the unit cube, so the prior is far
                   from flat in these coordinates and a bare corner plot cannot
                   separate "the data constrained this" from "the screens did".
  mr_band.png      M-R posterior band, all stable branches.
  lambda_band.png  Lambda(M) band and the Lambda_1.4 posterior.
  eos_band.png     P(eps) and cs^2(eps) bands, with the hadronic table drawn
                   through them for reference.
  dynesty_run.png  evidence/live-point diagnostics from the saved checkpoint.

Use compare.py for the cross-EOS ln B figure.
"""
import argparse
import collections
import json
import os

import numpy as np

import config
from hybrid_eos import P0, HybridEOS
from model import HybridModel
from priors import PARAM_LABELS, PARAM_NAMES

M_REF = 1.4            # the mass at which Lambda is quoted


def wquantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return np.interp(q, cw, x[o])


def screened_prior(model, n, seed):
    """Prior draws that survive model.screen(), i.e. the prior the sampler sees.

    Costs a full TOV solve per accepted draw, so n is kept small by default.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        th = model.prior_transform(rng.random(model.ndim))
        if model.screen(th)[0] == "accepted":
            out.append(th)
    return np.asarray(out)


def lambda_at(seq, m):
    """Lambda(m) on whichever stable branch contains m, else nan."""
    for b in seq.branches:
        if b["M"][0] <= m <= b["M"][-1]:
            return float(np.exp(b["interp"]["Lam"](m)))
    return np.nan


def eos_curve(eos, eps_grid):
    """P(eps) and cs^2(eps) in MeV/fm^3 along one hybrid EOS.

    cs2_at_P is a function of pressure, and pressure is constant across the
    Maxwell jump, so it cannot see the jump: it returns the value at the base of
    the polytrope for every eps inside it.  dP/deps is zero there, so the jump is
    zeroed explicitly -- otherwise the plot shows a stiff plateau exactly where
    the EOS is at its softest.
    """
    P = np.array([eos.Pbar_at_eps_central(e) for e in eps_grid])
    cs2 = np.array([eos.cs2_at_P(p) for p in P])
    cs2[(eps_grid > eos.eps_t * P0) & (eps_grid < eos.eps_s * P0)] = 0.0
    return P * P0, cs2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--ndraw", type=int, default=300)
    ap.add_argument("--nprior", type=int, default=300,
                    help="screened prior draws for the corner overlay; 0 to skip")
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
    if summ.get("prior_acceptance"):
        a = summ["prior_acceptance"]
        print(f"   prior acceptance f_acc = {a['f_acc']:.3f} +/- {a['f_acc_err']:.3f} "
              f"({a['n_accepted']}/{a['n_used']} draws): ln Z on the unit cube was "
              f"{summ['logZ_unit_cube']:.2f}, renormalising to the truncated prior "
              f"added {-a['ln_f_acc']:+.2f} nat")
    if summ.get("hadronic_components"):
        print("   hadronic baseline by term: " +
              "  ".join(f"{k}={v:+.2f}" for k, v in summ["hadronic_components"].items()))
    print(f"effective sample size: {1/np.sum(w**2):.0f}")
    print(f"{'param':>16} {'median':>9} {'68% low':>9} {'68% hi':>9} {'95% low':>9} {'95% hi':>9}")
    for i, n in enumerate(PARAM_NAMES):
        q = [wquantile(x[:, i], w, v) for v in (0.5, 0.16, 0.84, 0.025, 0.975)]
        print(f"{n:>16} {q[0]:9.2f} {q[1]:9.2f} {q[2]:9.2f} {q[3]:9.2f} {q[4]:9.2f}")
    r = x[:, 1] / x[:, 0]
    q = [wquantile(r, w, v) for v in (0.5, 0.16, 0.84)]
    print(f"{'d_eps_trans/eps_trans':>21} {q[0]:6.2f} (68%: {q[1]:.2f}-{q[2]:.2f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("plots skipped (matplotlib unavailable):", e)
        plt = None

    model = HybridModel(summ["eos"], None)
    written = []

    # ---- corner, with the screened prior behind the posterior ----------------
    if plt is not None:
        try:
            import corner
            rng_lo = x.min(0)
            rng_hi = x.max(0)
            pri = screened_prior(model, args.nprior, args.seed + 1) if args.nprior else None
            if pri is not None and len(pri):
                rng_lo = np.minimum(rng_lo, pri.min(0))
                rng_hi = np.maximum(rng_hi, pri.max(0))
            span = [(lo, hi) for lo, hi in zip(rng_lo, rng_hi)]
            fig = None
            if pri is not None and len(pri):
                # the prior as light points, not contours: a few hundred draws are
                # too sparse for contours and corner warns about it
                fig = corner.corner(pri, labels=PARAM_LABELS, range=span, color="0.6",
                                    plot_density=False, plot_contours=False,
                                    data_kwargs=dict(alpha=0.35, ms=2.5),
                                    hist_kwargs=dict(density=True))
            corner.corner(x, weights=w, labels=PARAM_LABELS, range=span, fig=fig,
                          color="C0", quantiles=[0.16, 0.5, 0.84], show_titles=True,
                          hist_kwargs=dict(density=True))
            if fig is not None:
                fig.text(0.72, 0.72, "grey: screened prior\nblue: posterior", fontsize=11)
            plt.savefig(os.path.join(args.outdir, "corner.png"), dpi=120)
            plt.close("all")
            written.append("corner.png")
        except Exception as e:
            print("corner plot skipped:", e)

    # ---- posterior-predictive diagnostics -------------------------------------
    import solver
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(x), args.ndraw, p=w)
    topo = collections.Counter()
    probe = hyb = n_ok = 0
    seidov = []
    Mtov, curves, eoss, nt, mu_t, lam_ref = [], [], [], [], [], []
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
        probe += Pc_max > eos.P_t           # some stable star contains polytrope matter
        hyb += seq.eps_c_tov > eos.params["eps_trans"]
        s = eos.seidov()
        seidov.append(s["ratio"] - s["threshold"])
        try:
            j = eos.number_density()["jumps"][0]
            nt.append(j["n_lo"] / config.N0)
            mu_t.append(j["mu"])
        except Exception:
            pass
        lam_ref.append(lambda_at(seq, M_REF))
        curves.append(seq)
        eoss.append(eos)
    n = max(n_ok, 1)
    print(f"\nposterior-predictive diagnostics over {n_ok} draws")
    print("  stable branches per draw:", {k: f"{v/n:.1%}" for k, v in sorted(topo.items())})
    print(f"  transition reached by some stable star: {probe/n:.1%}"
          "   <- draws below this are effectively hadronic models")
    print(f"  M_TOV star has a hybrid core (eps_c > eps_trans): {hyb/n:.1%}")
    sd = np.array(seidov)
    print(f"  Seidov margin at the transition (ratio - threshold): median {np.median(sd):+.3f}, "
          f"{100*np.mean(sd>0):.0f}% above the line (immediately unstable after the jump)")
    if nt:
        print(f"  transition: n_t = {np.median(nt):.2f} n0 "
              f"(68% {np.percentile(nt,16):.2f}-{np.percentile(nt,84):.2f}), "
              f"mu = {np.median(mu_t):.0f} MeV")
    print(f"  M_TOV: median {np.median(Mtov):.3f}, "
          f"68% {np.percentile(Mtov,16):.3f}-{np.percentile(Mtov,84):.3f}")
    lam = np.array(lam_ref, dtype=float)
    ok = np.isfinite(lam)
    if ok.any():
        print(f"  Lambda_{M_REF}: median {np.median(lam[ok]):.0f}, "
              f"68% {np.percentile(lam[ok],16):.0f}-{np.percentile(lam[ok],84):.0f} "
              f"({100*ok.mean():.0f}% of draws reach {M_REF} Msun)")

    if plt is None:
        return

    # ---- M-R band -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for s in curves[:150]:
        for b in s.branches:
            ax.plot(b["R"], b["M"], color="C0", alpha=0.08, lw=1)
    ax.set_xlabel("R [km]")
    ax.set_ylabel(r"M [$M_\odot$]")
    ax.set_xlim(8, 16)
    ax.set_ylim(0.8, 3.0)
    plt.savefig(os.path.join(args.outdir, "mr_band.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append("mr_band.png")

    # ---- Lambda(M) band and the Lambda_1.4 posterior -------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for s in curves[:150]:
        for b in s.branches:
            a1.plot(b["M"], b["Lam"], color="C0", alpha=0.08, lw=1)
    a1.set_yscale("log")
    a1.set_xlabel(r"M [$M_\odot$]")
    a1.set_ylabel(r"$\Lambda$")
    a1.set_xlim(1.0, 2.4)
    a1.axvline(M_REF, color="0.4", ls=":", lw=1)
    if ok.any():
        a2.hist(lam[ok], bins=30, color="C0", alpha=0.8, density=True)
        for v, ls in ((np.median(lam[ok]), "-"), (np.percentile(lam[ok], 16), "--"),
                      (np.percentile(lam[ok], 84), "--")):
            a2.axvline(v, color="k", ls=ls, lw=1)
    a2.set_xlabel(r"$\Lambda_{%.1f}$" % M_REF)
    a2.set_ylabel("posterior density")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "lambda_band.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append("lambda_band.png")

    # ---- EOS band: P(eps) and cs^2(eps) --------------------------------------
    eps_hi = max(float(np.max(s.eps_c_tov)) for s in curves) * 1.05
    grid = np.linspace(100.0, max(eps_hi, 400.0), 240)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for eos in eoss[:150]:
        P, cs2 = eos_curve(eos, grid)
        a1.plot(grid, P, color="C0", alpha=0.08, lw=1)
        a2.plot(grid, cs2, color="C0", alpha=0.08, lw=1)
    tab = model.table
    m = (tab.eps > grid[0]) & (tab.eps < grid[-1])
    a1.plot(tab.eps[m], tab.P[m], color="k", lw=1.5, label="hadronic table")
    a1.set_xlabel(r"$\varepsilon$ [MeV/fm$^3$]")
    a1.set_ylabel(r"P [MeV/fm$^3$]")
    a1.set_yscale("log")
    a1.legend(loc="lower right", fontsize=9)
    a2.axhline(1.0, color="r", ls="--", lw=1, label=r"$c_s^2=1$")
    a2.axhline(1 / 3, color="0.4", ls=":", lw=1, label=r"$c_s^2=1/3$")
    a2.set_xlabel(r"$\varepsilon$ [MeV/fm$^3$]")
    a2.set_ylabel(r"$c_s^2$")
    a2.set_ylim(0, 1.15)
    a2.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "eos_band.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append("eos_band.png")

    # ---- dynesty's own run diagnostics ---------------------------------------
    ckpt = os.path.join(args.outdir, "dynesty.save")
    if os.path.exists(ckpt):
        try:
            import dynesty
            from dynesty import plotting as dyplot
            res = dynesty.NestedSampler.restore(ckpt).results
            fig, _ = dyplot.runplot(res)
            plt.savefig(os.path.join(args.outdir, "dynesty_run.png"), dpi=110,
                        bbox_inches="tight")
            plt.close("all")
            fig, _ = dyplot.traceplot(res, labels=PARAM_LABELS, show_titles=True)
            plt.savefig(os.path.join(args.outdir, "dynesty_trace.png"), dpi=110,
                        bbox_inches="tight")
            plt.close("all")
            written += ["dynesty_run.png", "dynesty_trace.png"]
        except Exception as e:
            print("dynesty diagnostics skipped:", e)
    else:
        print("dynesty diagnostics skipped: no", ckpt)

    print("wrote " + ", ".join(written))


if __name__ == "__main__":
    main()
