"""
validate.py -- numerical checks of the solver, the EOS construction and the data.

    python validate.py [--quick] [--no-data]

 1. regression vs an independent scipy integration of the same TOV + y equations
 2. absolute check against published numbers for a real EOS (FSUGarnet)
 3. k2(beta, y): double precision vs mpmath
 4. first-order jump: sharp (Takatsy-Kovacs Eq. 11 / original PPL) vs a thin
    smoothed transition
 5. convergence of M_TOV, R_1.4, Lambda_1.4 in rtol and in the number of stars,
    and -- the thing that actually used to be wrong -- grid robustness of the
    stable-branch decomposition
 6. causality screen: the sub-sampled envelope vs a brute-force scan
 7. Maxwell construction: mu continuity, monotone n, the first law
 8. data: headline reproduction, ESS, KDE bandwidth and split-half stability,
    and the GW170817 sampling-prior claims (flat Lambda, non-flat q)
 9. prior statistics, Seidov consistency, branch-rule sensitivity, timing
"""
import collections
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

import config
import datasets as D
import solver
from hybrid_eos import P0, R0, HadronicTable, HybridEOS
from model import HybridModel

QUICK = "--quick" in sys.argv
NODATA = "--no-data" in sys.argv
EOS = config.HADRONIC_EOS_FILES[0]
tab = HadronicTable(EOS, config.EOS_SKIPROWS, config.EOS_COLS)
sw = config.SWEEP

# A causal hybrid EOS with a genuine stable hybrid core, used by tests 4, 5 and 7.
# (The parameter vector these tests used to run on had cs^2 = 4.0 inside the star,
# so it could never appear in the posterior and told us nothing about the region
# the sampler actually visits.)
CASE = dict(eps_trans=450.0, d_eps_trans=0.35 * 450.0, eps_1=900.0, d_eps_1=120.0,
            gamma1=1.8, gamma2=2.2)
FAILURES = []


def hdr(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def check(ok, label):
    print(("   PASS  " if ok else "   FAIL  ") + label)
    if not ok:
        FAILURES.append(label)


def case_eos():
    return HybridEOS(tab, CASE["eps_trans"], CASE["d_eps_trans"], CASE["eps_1"],
                     CASE["d_eps_1"], CASE["gamma1"], CASE["gamma2"])


def k2_mp(beta, y, dps=50):
    import mpmath as mp
    with mp.workdps(dps):
        b, yy = mp.mpf(beta), mp.mpf(y)
        t1 = 2 - yy + 2 * b * (yy - 1)
        br = (2 * b * (6 - 3 * yy + 3 * b * (5 * yy - 8) + 2 * b**2 *
                       (13 - 11 * yy + b * (3 * yy - 2) + 2 * b**2 * (1 + yy)))
              + 3 * (1 - 2 * b)**2 * t1 * mp.log(1 - 2 * b))
        return float(mp.mpf(8) / 5 * b**5 * (1 - 2 * b)**2 * t1 / br)


# ---------------------------------------------------------------- 1
def ref_star(Pc, rtol=1e-9):
    e_, p_ = tab.eps / P0, tab.P / P0
    lE = PchipInterpolator(np.log(p_), np.log(e_), extrapolate=False)
    dl = lE.derivative(1)
    Pmin = p_.min()

    def TOV(R, y):
        M, P, yL = y
        Pe = max(P, Pmin); lp = np.log(Pe); e = np.exp(lE(lp)); s = dl(lp)
        cs2 = (Pe / e) / s if s > 0 else 1e10
        dM = e * R**2
        dP = -(M * e / (2 * R**2)) * (1 + P / e) * (1 + R**3 * P / M) / (1 - M / R)
        EL = 1 / (1 - M / R); yt = 1 + .5 * R**2 * (P - e); Q = 5 * e + 9 * P + (e + P) / cs2
        r2Q = .5 * R**2 * EL * Q - 6 * EL - (EL**2 / R**2) * (M + R**3 * P)**2
        return [dM, dP, -(1 / R) * (yL**2 + yL * EL * yt + r2Q)]

    ev = lambda R, y: y[1]; ev.terminal = True; ev.direction = -1
    ec = np.exp(lE(np.log(Pc))); r0 = 1e-6
    sol = solve_ivp(TOV, (r0, 5000), [ec * r0**3 / 3, Pc, 2.0], method="RK45",
                    events=ev, rtol=rtol, atol=1e-10)
    Rf = sol.t_events[0][0]; M, P, y = sol.y_events[0][0]; beta = M / (2 * Rf)
    return Rf * R0, M, (2 / 3) * k2_mp(beta, y) / beta**5


def test_regression():
    hdr("1. regression vs an independent scipy integration (pure hadronic)")
    eos = HybridEOS.pure_hadronic(tab)
    print(f"{'eps_c':>7} {'M':>7} {'dR/R':>11} {'dM/M':>11} {'dLambda/Lambda':>15}")
    worst = worst_low = 0.0
    m_min = config.LIKELIHOOD["mass_prior_min"]
    for ec in ([200, 600, 1100] if QUICK else [150, 300, 600, 900, 1200]):
        Pc = float(tab.P_bar_of_eps_bar(ec / P0)); r = ref_star(Pc)
        o = solver.sweep_kernel(np.array([Pc]), eos.par, tab.x, tab.y, tab.d, tab.Pbar_min,
                                sw["rtol"], sw["atol_M"], sw["atol_P"], sw["atol_y"])[0]
        Lam = (2 / 3) * o[6] / o[4]**5
        d = (o[2] / r[0] - 1, o[3] / r[1] - 1, Lam / r[2] - 1)
        if o[3] >= m_min:
            worst = max(worst, abs(d[1]), abs(d[2]))
        else:
            worst_low = max(worst_low, abs(d[1]), abs(d[2]))
        print(f"{ec:7d} {o[3]:7.3f} {d[0]:+11.2e} {d[1]:+11.2e} {d[2]:+15.2e}")
    print("   The residual is dominated by the surface definition: the reference integrates")
    print("   to P = 0, this solver to the table's lowest tabulated pressure.  It is largest")
    print("   for the puffy low-mass stars, which the likelihood never uses (the population")
    print(f"   prior starts at m = {m_min}), so the tolerance below is applied above that mass;")
    print(f"   worst discrepancy among the lighter stars, for the record: {worst_low:.1e}")
    check(worst < 3e-4, f"M and Lambda agree with the reference to {worst:.1e} above m = {m_min}")


# ---------------------------------------------------------------- 2
def test_published():
    hdr("2. absolute check against published numbers")
    ref = {"EOSBetaFSUGarnet": dict(Lam14=(623, 15), Mtov=(2.07, 0.05),
                                    src="Piekarewicz & Fattoyev 2019 PRC 99 045802; "
                                        "Chen & Piekarewicz 2015 (M_max = 2.07 +/- 0.02)")}
    for f, r in ref.items():
        t = HadronicTable(os.path.join(os.path.dirname(EOS) or ".", f + ".dat"),
                          config.EOS_SKIPROWS, config.EOS_COLS)
        s = solver.compute_sequence(HybridEOS.pure_hadronic(t))
        L14 = float(s.branch_values(np.array([1.4]), "Lam")[0, 0])
        print(f"   {f}: Lambda_1.4 = {L14:.1f} (published {r['Lam14'][0]}), "
              f"M_TOV = {s.M_tov:.3f} (published {r['Mtov'][0]})")
        print(f"     {r['src']}")
        check(abs(L14 - r["Lam14"][0]) < r["Lam14"][1], f"{f} Lambda_1.4 matches publication")
        check(abs(s.M_tov - r["Mtov"][0]) < r["Mtov"][1], f"{f} M_TOV matches publication")


# ---------------------------------------------------------------- 3
def test_k2():
    hdr("3. k2: double precision (numba) vs mpmath (50 digits)")
    worst = 0.0
    for beta in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3):
        err = max(abs(solver.k2_love(beta, y) / k2_mp(beta, y) - 1) for y in (0.5, 1.0, 2.0, 3.0))
        worst = max(worst, err)
        print(f"   beta={beta:5.3f}  max rel err = {err:.1e}")
    check(worst < 1e-6, f"double precision k2 good to {worst:.1e} for beta >= 0.005")


# ---------------------------------------------------------------- 4
def test_jump():
    hdr("4. first-order jump: sharp (TK Eq.11 / original PPL) vs a thin smoothed transition")
    eos = case_eos()
    g1 = CASE["gamma1"]

    def smooth(P, w):
        z = np.clip((P - eos.P_t) / w, -700, 700); s = 1 / (1 + np.exp(-z)); ds = s * (1 - s) / w
        eh, ep = float(tab.eps_bar(P)), (P / eos.K1)**(1 / g1); dP = 1e-6 * max(P, 1e-12)
        dh = (float(tab.eps_bar(P + dP)) - float(tab.eps_bar(P - dP))) / (2 * dP)
        dp = ep / (g1 * P)
        return (1 - s) * eh + s * ep, 1.0 / (ds * (ep - eh) + (1 - s) * dh + s * dp)

    def ref(Pc, w):
        Pmin = tab.Pbar_min

        def f(R, y):
            M, P, yL = y; e, cs2 = smooth(max(P, Pmin), w)
            dM = e * R**2
            dP = -(M * e / (2 * R**2)) * (1 + P / e) * (1 + R**3 * P / M) / (1 - M / R)
            EL = 1 / (1 - M / R); yt = 1 + .5 * R**2 * (P - e); Q = 5 * e + 9 * P + (e + P) / cs2
            r2Q = .5 * R**2 * EL * Q - 6 * EL - (EL**2 / R**2) * (M + R**3 * P)**2
            return [dM, dP, -(1 / R) * (yL**2 + yL * EL * yt + r2Q)]

        ev = lambda R, y: y[1] - Pmin; ev.terminal = True; ev.direction = -1
        e0, _ = smooth(Pc, w); r0 = 1e-6
        sol = solve_ivp(f, (r0, 50), [e0 * r0**3 / 3, Pc, 2.0], method="LSODA", events=ev,
                        rtol=1e-10, atol=1e-12)
        Rf = sol.t_events[0][0]; M, P, y = sol.y_events[0][0]; beta = M / (2 * Rf)
        return M, y, (2 / 3) * solver.k2_love(beta, y) / beta**5

    # the smoothed reference only blends the FIRST transition, so the test star
    # must not reach the second one: keep Pc strictly between P_t and P_1
    Pc = float(np.sqrt(eos.P_t * eos.P_1))
    assert eos.P_t < Pc < eos.P_1
    print(f"   test star: Pc/P_t = {Pc/eos.P_t:.2f}, Pc/P_1 = {Pc/eos.P_1:.2f}")
    sharp = {}
    for mode, lab in ((0, "sharp, corrected TK Eq.(11)"), (1, "sharp, original PPL (no P term)")):
        par = eos.par.copy(); par[8] = mode
        o = solver.sweep_kernel(np.array([Pc]), par, tab.x, tab.y, tab.d, tab.Pbar_min,
                                1e-8, 1e-10, 1e-10, 1e-10)[0]
        sharp[mode] = (o[5], (2 / 3) * o[6] / o[4]**5)
        print(f"   {lab:34s} y_R = {o[5]:.5f}  Lambda = {sharp[mode][1]:.3f}")
    last = None
    for w in ((1e-3,) if QUICK else (1e-2, 1e-3, 1e-4)):
        M, y, L = ref(Pc, w * eos.P_t)
        last = (y, L)
        print(f"   smoothed, width = {w:g} P_t{'':14s} y_R = {y:.5f}  Lambda = {L:.3f}")
    print("   -> original PPL form differs by %.1f%% in y_R and %.2f%% in Lambda"
          % (100 * (sharp[1][0] / sharp[0][0] - 1), 100 * (sharp[1][1] / sharp[0][1] - 1)))
    check(abs(last[1] / sharp[0][1] - 1) < 2e-3,
          "thin smoothed transition converges to the corrected TK jump, not to PPL")


# ---------------------------------------------------------------- 5
def test_convergence():
    hdr("5. convergence and grid robustness")
    eos = case_eos()

    def summarize(cfg):
        s = solver.compute_sequence(eos, cfg)
        return (s.M_tov, s.branch_values(np.array([1.4]), "R")[0, 0],
                s.branch_values(np.array([1.4]), "Lam")[0, 0], s.n_points, s.n_branches)

    ref_cfg = dict(sw, rtol=1e-9, n_base=400, refine_rounds=4)
    r0 = summarize(ref_cfg)
    print(f"   reference (rtol=1e-9, n_base=400): M_TOV={r0[0]:.5f} R1.4={r0[1]:.4f} "
          f"Lambda1.4={r0[2]:.3f} branches={r0[4]}")
    worst = 0.0
    for rt, nb in ((1e-5, 70), (1e-6, 70), (1e-6, 40), (1e-7, 70), (1e-6, 110)):
        r = summarize(dict(sw, rtol=rt, n_base=nb))
        worst = max(worst, abs(r[0] - r0[0]))
        print(f"   rtol={rt:g} n_base={nb:3d}: dM_TOV={r[0]-r0[0]:+.1e}  dR1.4={r[1]-r0[1]:+.1e} km"
              f"  dLambda1.4/Lambda={r[2]/r0[2]-1:+.1e}  ({r[3]} stars, {r[4]} branches)")
    check(worst < 1e-4, f"M_TOV converged to {worst:.1e} at the production settings")

    # branch decomposition must not depend on the grid.  This is the test that the
    # previous turning-point counter failed: refinement eventually bracketed every
    # extremum below the ODE noise floor, the sign of dM then flipped at random,
    # and real branches disappeared (M_TOV once moved by a factor of three).
    print("\n   branch decomposition across three very different grids:")
    from priors import PriorTransform
    pt = PriorTransform(tab)
    rng = np.random.default_rng(3)
    grids = [sw,
             dict(sw, n_base=250, refine_rounds=3, rtol=1e-7),
             dict(sw, n_base=600, refine_rounds=4, rtol=1e-8)]
    n_multi = n_bad = n_ok = 0
    target = 20 if QUICK else 60
    tries = 0
    # multi-branch EOSs are only a few percent of the prior but are the whole point
    # of the twin-star part of this study, so keep drawing until some turn up
    while (n_ok < target or n_multi < (2 if QUICK else 5)) and tries < 4000:
        tries += 1
        th = pt(rng.uniform(size=6)); e = HybridEOS(tab, *th)
        if not e.valid or th[4] * e.P_t / e.eps_s1 > 1 + 1e-6:
            continue
        ss = [solver.compute_sequence(e, g) for g in grids]
        if not all(s.ok for s in ss) or ss[0].cs2_max > 1 + 1e-6:
            continue
        n_ok += 1
        nb = {s.n_branches for s in ss}
        mt = [s.M_tov for s in ss]
        if ss[0].n_branches > 1:
            n_multi += 1
        if len(nb) > 1 or max(mt) - min(mt) > 1e-3:
            n_bad += 1
            print(f"      disagreement: branches {[s.n_branches for s in ss]}  M_TOV {mt}")
    print(f"   {n_ok} accepted EOSs ({n_multi} multi-branch), {n_bad} grid disagreements")
    check(n_bad == 0, "stable-branch decomposition and M_TOV are grid independent")


# ---------------------------------------------------------------- 6
def test_causality_envelope():
    hdr("6. causality screen: sub-sampled envelope vs a brute-force scan")
    x = tab.x
    xs = np.unique(np.concatenate([np.linspace(x[i], x[i + 1], 257)[:-1]
                                   for i in range(len(x) - 1)] + [x[-1:]]))
    k = np.clip(np.searchsorted(x, xs) - 1, 0, len(x) - 2)
    h = x[k + 1] - x[k]; t = (xs - x[k]) / h; t2, t3 = t * t, t * t * t
    ly = ((2 * t3 - 3 * t2 + 1) * tab.y[k] + (t3 - 2 * t2 + t) * h * tab.d[k]
          + (-2 * t3 + 3 * t2) * tab.y[k + 1] + (t3 - t2) * h * tab.d[k + 1])
    dly = ((6 * t2 - 6 * t) / h * tab.y[k] + (3 * t2 - 4 * t + 1) * tab.d[k]
           + (-6 * t2 + 6 * t) / h * tab.y[k + 1] + (3 * t2 - 2 * t) * tab.d[k + 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        cs2 = np.where(dly > 0, np.exp(xs - ly) / dly, np.inf)
    brute = np.maximum.accumulate(cs2)
    env = np.array([tab.cs2_max_below(np.exp(v)) for v in xs])   # as the screen is actually called
    knots = np.maximum.accumulate(np.where(tab.d > 0, np.exp(tab.x - tab.y) / tab.d, np.inf))
    knot_at = knots[np.clip(np.searchsorted(tab.x, xs, side="right") - 1, 0, None)]
    ratio_env = np.nanmax(brute / np.maximum(env, 1e-30))
    ratio_knot = np.nanmax(brute[np.isfinite(knot_at)] / np.maximum(knot_at[np.isfinite(knot_at)], 1e-30))
    print(f"   brute force (256 sub-samples/interval) / envelope: max ratio {ratio_env:.5f}")
    print(f"   brute force / knots only (the old screen)                   : max ratio {ratio_knot:.4f}")
    print(f"   max cs^2 anywhere in {EOS}: {brute[-1]:.5f}")
    check(ratio_env < 1.001, "the causality screen under-reports cs^2 by < 0.1%")


# ---------------------------------------------------------------- 7
def test_maxwell():
    hdr("7. Maxwell construction: chemical potential, baryon density, first law")
    eos = case_eos()
    ok, rep = eos.maxwell_report()
    print("   " + rep.replace("\n", "\n   "))
    d = eos.number_density()
    for i, j in enumerate(d["jumps"], 1):
        mu_lo = (j["eps"] + j["P"]) / j["n_lo"]
        mu_hi = (j["eps"] + j["d_eps"] + j["P"]) / j["n_hi"]
        check(abs(mu_hi / mu_lo - 1) < 1e-12, f"mu continuous across jump {i}")
        check(j["d_n"] >= 0, f"Delta n >= 0 across jump {i}")
    e, P, n = d["eps"], d["P"], d["n"]
    m = (e > eos.eps_s1 * P0 * 1.05) & (e < eos.eps_1b * P0 * 0.95)
    x = 1 / n[m]; y = e[m] / n[m]
    res = np.max(np.abs(np.gradient(y, x) + P[m]) / P[m])
    print(f"   first law d(eps/n)/d(1/n) = -P inside polytrope 1: max residual {res:.2e}")
    check(res < 1e-2, "first law satisfied inside the polytrope")
    check(ok, "n and mu monotone above the outer crust")


# ---------------------------------------------------------------- 8
def test_data():
    hdr("8. observational data: headlines, ESS, bandwidth, and the GW sampling prior")
    from likelihood import GWTerm, Likelihood, MassRadiusTerm
    like = Likelihood(verbose=True)
    print()
    print(like.describe())
    seq = solver.compute_sequence(HybridEOS.pure_hadronic(tab))

    print("\n   ln L stability under bandwidth and a split-half of the samples:")
    for t in like.mr_terms:
        sp = config.DATA["mr_sources"][t.name]
        ss = t.samples
        perm = np.random.default_rng(1).permutation(len(ss.x)); a, b = perm[::2], perm[1::2]
        out = []
        for bw in ((0.3, 1.0) if QUICK else (0.15, 0.3, 0.6, 1.0)):
            kk = D.GridKDE(ss.x, ss.w, bw_factor=bw, bounds=sp.get("bounds", ()))
            T = MassRadiusTerm(t.name, ss, kk, t.m_min, t.n_grid); T.rule = like.rule
            ka = D.GridKDE(ss.x[a], ss.w[a], bw_factor=bw, bounds=sp.get("bounds", ()))
            kb = D.GridKDE(ss.x[b], ss.w[b], bw_factor=bw, bounds=sp.get("bounds", ()))
            Ta = MassRadiusTerm("a", ss, ka, t.m_min, t.n_grid); Ta.rule = like.rule
            Tb = MassRadiusTerm("b", ss, kb, t.m_min, t.n_grid); Tb.rule = like.rule
            out.append((bw, np.log(T(seq)), np.log(Ta(seq)) - np.log(Tb(seq))))
        spread = max(o[1] for o in out) - min(o[1] for o in out)
        print("   %-8s " % t.name + "  ".join(f"bw={o[0]:.2f} lnL={o[1]:+.3f} (halves {o[2]:+.3f})"
                                              for o in out))
        check(spread < 0.3, f"{t.name} ln L varies by < 0.3 over a 7x bandwidth range")

    if like.gw_term is not None:
        g = like.gw_term; ss = g.samples
        out = []
        for bw in ((1.0,) if QUICK else (0.6, 1.0, 1.5)):
            kk = D.GridKDE(ss.x, ss.w, bw_factor=bw, bounds=config.DATA["gw"]["bounds"])
            T = GWTerm("gw", ss, kk, g.Mc, g.m_min, g.n_grid, g.population, g.q_min)
            T.rule = like.rule
            out.append((bw, np.log(T(seq))))
        print("   %-8s " % g.name + "  ".join(f"bw={o[0]:.1f} lnL={o[1]:+.3f}" for o in out))
        for n in (60, 120, 240):
            T = GWTerm("gw", ss, g.kde, g.Mc, g.m_min, n, g.population, g.q_min)
            T.rule = like.rule
            print(f"      n_q_grid={n:4d}: ln L = {np.log(T(seq)):+.4f}")

        # the sampling-prior claims this term depends on
        import h5py
        with h5py.File(config.DATA["gw"]["file"], "r") as f:
            pr = f["IMRPhenomPv2NRT_lowSpin_prior"][:]
        for key in ("lambda1", "lambda2"):
            h, _ = np.histogram(pr[key].astype(float), bins=20, range=(0, 5000))
            rel = h.std() / h.mean()
            print(f"      prior {key}: flat on [0,5000] to {rel:.1%} (Poisson floor "
                  f"{1/np.sqrt(h.mean()):.1%})")
            check(rel < 3 / np.sqrt(h.mean()), f"{key} sampling prior is flat")
        m1, m2 = pr["m1_detector_frame_Msun"].astype(float), pr["m2_detector_frame_Msun"].astype(float)
        Mc = D.chirp_mass(m1, m2); q = m2 / m1
        sel = np.abs(Mc / np.median(Mc[np.abs(Mc - 1.1976) < 0.12]) - 1) < 0.10
        h, e = np.histogram(q[sel], bins=16, range=(0.4, 1.0))
        c = .5 * (e[1:] + e[:-1]); pred = D.prior_divide({"kind": "q_from_uniform_m"}, c)
        pred = pred * h.sum() / pred.sum()
        chi2 = np.sum((h - pred) ** 2 / np.maximum(h, 1)) / len(h)
        print(f"      prior q at fixed Mc vs q^(-6/5)(1+q)^(2/5): chi2/dof = {chi2:.2f} "
              f"(N={sel.sum()})")
        check(chi2 < 3.0, "q sampling prior matches the uniform-component-mass form")
        print("      NOTE Lambda-tilde from these samples: median %.0f, 90%% upper %.0f."
              % _lambda_tilde(ss))
        print("           The LVC headline (<= 720) is quoted after reweighting to a flat")
        print("           Lambda-tilde prior; this pipeline deliberately uses the raw")
        print("           posterior with its flat Lambda1, Lambda2 prior divided out.")


def _lambda_tilde(ss):
    q, L1, L2 = ss.x.T
    m1, m2 = D.masses_from_Mc_q(ss.Mc, q)
    Lt = 16. / 13. * ((m1 + 12 * m2) * m1**4 * L1 + (m2 + 12 * m1) * m2**4 * L2) / (m1 + m2)**5
    return float(np.median(Lt)), float(np.percentile(Lt, 90))


# ---------------------------------------------------------------- 9
def test_prior_stats():
    hdr("9. prior statistics, Seidov consistency, branch-rule sensitivity, timing")
    import mockdata
    from likelihood import Likelihood
    eos_h = HybridEOS.pure_hadronic(tab)
    seq_h = solver.compute_sequence(eos_h)
    likes = {r: mockdata.make_mock_likelihood(seq_h, seed=0,
                                              like_cfg=dict(config.LIKELIHOOD, branch_rule=r))
             for r in ("mixture", "max", "sum")}
    model = HybridModel(EOS, likes["mixture"])
    n = 80 if QUICK else 400
    rng = np.random.default_rng(7)
    reasons = collections.Counter(); nbr = collections.Counter()
    seid_ok = seid_n = 0; multi = []
    t0 = time.time()
    for _ in range(n):
        th = model.prior_transform(rng.uniform(size=6))
        o = model.evaluate(th)
        reasons[o["reason"][:28] or "accepted"] += 1
        if o["logL"] > -np.inf:
            s = o["seq"]; nbr[s.n_branches] += 1
            if s.n_branches > 1:
                multi.append(th)
            e = o["eos"]; j = np.searchsorted(s.Pc, e.P_t); sd = e.seidov()[0]
            if (2 <= j < len(s.Pc) - 2 and s.Pc[j - 1] < e.P_t < s.Pc[j]
                    and (s.M[j - 1] - s.M[j - 2]) / s.M[j - 1] > 2e-4
                    and abs(sd["ratio"] - sd["threshold"]) > 0.02):
                seid_n += 1
                seid_ok += (s.M[j + 1] < s.M[j]) == sd["unstable"]
    dt = (time.time() - t0) / n
    print(f"   {n} prior draws: {dict(reasons)}")
    acc = sum(nbr.values())
    print(f"   accepted by number of stable branches: {dict(nbr)}  "
          f"(multi-branch fraction = {sum(v for k, v in nbr.items() if k > 1)/max(acc,1):.1%})")
    if seid_n:
        print(f"   Seidov criterion vs computed dM/dPc at the first transition: "
              f"{seid_ok}/{seid_n} consistent")
    print(f"   mean wall time per likelihood evaluation: {dt:.3f} s")
    # the sampler needs a workable acceptance rate; below ~5% dynesty's initial
    # live-point generation gets expensive.  The eps_1 cap costs about 8 points.
    check(acc > 0.10 * n, f"prior acceptance {acc/n:.1%} is usable for nested sampling")
    if multi:
        d = []
        for th in multi[:40]:
            ll = {r: HybridModel(EOS, likes[r]).evaluate(th)["logL"] for r in likes}
            d.append((ll["max"] - ll["mixture"], ll["sum"] - ll["mixture"]))
        d = np.array(d)
        print(f"   branch-rule sensitivity on {len(d)} multi-branch EOSs (synthetic data): "
              f"median lnL(max)-lnL(mixture) = {np.median(d[:,0]):+.2f} "
              f"(max {d[:,0].max():+.2f}); median lnL(sum)-lnL(mixture) = {np.median(d[:,1]):+.2f}")


if __name__ == "__main__":
    test_regression(); test_published(); test_k2(); test_jump()
    test_convergence(); test_causality_envelope(); test_maxwell()
    if not NODATA:
        test_data()
    test_prior_stats()
    hdr("SUMMARY")
    if FAILURES:
        print("   %d FAILED:" % len(FAILURES))
        for f in FAILURES:
            print("     -", f)
        sys.exit(1)
    print("   all checks passed")
