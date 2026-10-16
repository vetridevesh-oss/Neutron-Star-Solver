"""
hybrid_eos.py -- hadronic table + two first-order transitions + two polytropes.

    P < P_t           : hadronic table (log-log Pchip, identical to your solver)
    P = P_t           : jump  eps_trans -> eps_trans + d_t         (Maxwell construction)
    P_t < P < P_1     : polytrope 1,  P = K1 eps^g1,  eps in [eps_trans+d_t, eps_1]
    P = P_1           : jump  eps_1 -> eps_1 + d_1
    P > P_1           : polytrope 2,  P = K2 eps^g2,  eps >= eps_1 + d_1

with P_t = P_hadronic(eps_trans), P_1 = K1 eps_1^g1,
K1 = P_t/(eps_trans+d_t)^g1 and K2 = P_1/(eps_1+d_1)^g2  (pressure continuous).
Speed of sound in a polytrope: cs^2 = g P/eps.
"""
import numpy as np
from scipy.constants import G, c, pi
from scipy.interpolate import PchipInterpolator

Msol = 1.98847e30                                           # kg
R0 = 2 * G * Msol / (c ** 2 * 1000)                          # km
P0 = c ** 8 / (32 * pi * G ** 3 * Msol ** 2) / 1.602176634e32  # MeV/fm^3  (units of eps-bar, P-bar)

BIG = 1e300  # "no transition" pressure
NSUB_CS2 = 64  # sub-samples per table interval for the cs^2 envelope


class HadronicTable:
    """Hadronic EOS in the dimensionless variables of the solver (eps-bar = eps/P0)."""

    def __init__(self, fname, skiprows=4, cols=(0, 1, 2)):
        tab = np.loadtxt(fname, skiprows=skiprows)
        e, p, n = tab[:, cols[0]], tab[:, cols[1]], tab[:, cols[2]]
        if not (np.all(np.diff(p) > 0) and np.all(np.diff(e) > 0)):
            raise ValueError("Hadronic table must have strictly increasing P and eps")
        self.fname = fname
        self.eps, self.P, self.n = e, p, n                     # MeV/fm^3, MeV/fm^3, fm^-3
        self.x = np.log(p / P0)                                # log P-bar
        self.y = np.log(e / P0)                                # log eps-bar
        pch = PchipInterpolator(self.x, self.y, extrapolate=False)
        self.d = pch.derivative(1)(self.x)                     # exact Pchip slopes at knots
        self.logP_of_logeps = PchipInterpolator(self.y, self.x, extrapolate=False)
        self.Pbar_min = float(np.exp(self.x[0]))
        self.Pbar_max = float(np.exp(self.x[-1]))
        # cs^2 at knots (exact derivative of the interpolant at a knot)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.cs2_knots = np.exp(self.x - self.y) / self.d
        # Running max of cs^2, evaluated on a SUB-SAMPLED grid.  A cubic Hermite
        # segment can have a smaller d(ln eps)/d(ln P) -- hence a larger cs^2 --
        # strictly inside an interval than at either knot, so a knot-only maximum
        # underestimates the true max (by up to ~50% on these tables) and would let
        # an acausal hadronic EOS through the screen.
        self._build_cs2_envelope(sub=NSUB_CS2)

    # -- numpy evaluation of the Pchip interpolant (same cubic Hermite as numba) --
    def eps_bar(self, Pbar):
        x = np.log(np.maximum(Pbar, self.Pbar_min))
        k = np.clip(np.searchsorted(self.x, x) - 1, 0, len(self.x) - 2)
        h = self.x[k + 1] - self.x[k]
        t = (x - self.x[k]) / h
        h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
        h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
        ly = h00 * self.y[k] + h10 * h * self.d[k] + h01 * self.y[k + 1] + h11 * h * self.d[k + 1]
        return np.exp(ly)

    def P_bar_of_eps_bar(self, ebar):
        return np.exp(self.logP_of_logeps(np.log(ebar)))

    def eps_of_n(self, n):
        """Energy density [MeV/fm^3] at number density n [fm^-3]."""
        return float(np.interp(n, self.n, self.eps))

    def eps_range_from_prior(self, spec):
        if spec["kind"] == "n0":
            from config import N0
            return self.eps_of_n(spec["lo"] * N0), self.eps_of_n(spec["hi"] * N0)
        return float(spec["lo"]), float(spec["hi"])

    def _build_cs2_envelope(self, sub=8):
        x = self.x
        xs = np.unique(np.concatenate(
            [np.linspace(x[i], x[i + 1], sub + 1)[:-1] for i in range(len(x) - 1)] + [x[-1:]]))
        k = np.clip(np.searchsorted(x, xs) - 1, 0, len(x) - 2)
        h = x[k + 1] - x[k]
        t = (xs - x[k]) / h
        t2, t3 = t * t, t * t * t
        ly = ((2 * t3 - 3 * t2 + 1) * self.y[k] + (t3 - 2 * t2 + t) * h * self.d[k]
              + (-2 * t3 + 3 * t2) * self.y[k + 1] + (t3 - t2) * h * self.d[k + 1])
        dly = ((6 * t2 - 6 * t) / h * self.y[k] + (3 * t2 - 4 * t + 1) * self.d[k]
               + (-6 * t2 + 6 * t) / h * self.y[k + 1] + (3 * t2 - 2 * t) * self.d[k + 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            cs2 = np.exp(xs - ly) / dly
        cs2 = np.where(np.isfinite(cs2) & (dly > 0), cs2, np.inf)   # dly<=0 => eps not increasing
        self._cs2_x = xs
        self.cs2_cummax = np.maximum.accumulate(cs2)

    def cs2_max_below(self, Pbar):
        """Max of cs^2 over every pressure in the table at or below Pbar.

        Uses the sub-sampled envelope built in __init__, not just the knots, and
        rounds the index UP so the sub-interval containing Pbar is included.  A
        causality screen must never under-report, and rounding down would miss
        whatever the interpolant does between the last sub-sample and Pbar.
        """
        k = np.searchsorted(self._cs2_x, np.log(Pbar), side="right")
        return float(self.cs2_cummax[min(max(k, 0), len(self.cs2_cummax) - 1)])

    # ---------------- baryon number density (Maxwell bookkeeping) ----------------
    def n_of_eps(self, eps):
        """Baryon number density [fm^-3] at energy density eps [MeV/fm^3], from the table."""
        return float(np.interp(eps, self.eps, self.n))

    def mu_of_eps(self, eps):
        """Baryon chemical potential mu = (eps + P)/n [MeV] on the hadronic table."""
        P = float(np.interp(eps, self.eps, self.P))
        return (eps + P) / self.n_of_eps(eps)


class HybridEOS:
    """Container for one parameter vector; `par` is the array handed to numba."""

    def __init__(self, table, eps_trans, d_eps_trans, eps_1, d_eps_1, gamma1, gamma2):
        self.table = table
        self.params = dict(eps_trans=eps_trans, d_eps_trans=d_eps_trans, eps_1=eps_1,
                           d_eps_1=d_eps_1, gamma1=gamma1, gamma2=gamma2)
        self.valid, self.reason = True, ""
        e_t = eps_trans / P0
        if not (table.eps[0] < eps_trans < table.eps[-1]):
            return self._fail("eps_trans outside table")
        if d_eps_trans < 0 or d_eps_1 < 0:
            return self._fail("negative jump")
        if gamma1 <= 0 or gamma2 <= 0:
            return self._fail("gamma<=0")
        start1 = eps_trans + d_eps_trans
        if eps_1 <= start1:
            return self._fail("eps_1 <= eps_trans + d_eps_trans")
        # transition pressure and the *actual* hadronic eps at it (consistent jump size)
        self.P_t = float(table.P_bar_of_eps_bar(e_t))
        self.eps_t = float(table.eps_bar(self.P_t))                    # eps-bar
        self.eps_s1 = self.eps_t + d_eps_trans / P0                    # bottom of polytrope 1
        self.K1 = self.P_t / self.eps_s1 ** gamma1
        self.eps_1b = eps_1 / P0
        self.P_1 = self.K1 * self.eps_1b ** gamma1
        self.eps_s2 = self.eps_1b + d_eps_1 / P0
        self.K2 = self.P_1 / self.eps_s2 ** gamma2
        self.d_t, self.d_1 = d_eps_trans / P0, d_eps_1 / P0
        if not (self.P_t < self.P_1 < BIG):
            return self._fail("P_t >= P_1")
        self.par = np.array([self.P_t, self.P_1, self.K1, gamma1, self.K2, gamma2,
                             self.d_t, self.d_1, 0.0], dtype=np.float64)

    @classmethod
    def pure_hadronic(cls, table):
        o = cls.__new__(cls)
        o.table, o.params, o.valid, o.reason = table, {}, True, ""
        o.P_t, o.P_1 = BIG, BIG
        o.par = np.array([BIG, BIG, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        o.is_hadronic = True
        return o

    def _fail(self, why):
        self.valid, self.reason = False, why

    # ---------------- numpy helpers ----------------
    def Pbar_at_eps_central(self, eps_c):
        """Pressure-bar of the star whose central energy density is eps_c [MeV/fm^3]."""
        e = eps_c / P0
        if getattr(self, "is_hadronic", False) or e <= self.eps_t:
            return float(self.table.P_bar_of_eps_bar(min(e, self.table.eps[-1] / P0 * 0.9999)))
        if e < self.eps_s1:
            return self.P_t
        if e <= self.eps_1b:
            return self.K1 * e ** self.params["gamma1"]
        if e < self.eps_s2:
            return self.P_1
        return self.K2 * e ** self.params["gamma2"]

    def cs2_at_P(self, Pbar):
        """cs^2 on the *occupied* branch of the EOS at pressure Pbar (scalar)."""
        if Pbar < self.P_t:
            lx = np.log(max(Pbar, self.table.Pbar_min))
            k = min(max(np.searchsorted(self.table.x, lx) - 1, 0), len(self.table.x) - 2)
            return float(self.table.cs2_knots[k])  # coarse; only used for diagnostics
        if Pbar < self.P_1:
            e = (Pbar / self.K1) ** (1 / self.params["gamma1"])
            return self.params["gamma1"] * Pbar / e
        e = (Pbar / self.K2) ** (1 / self.params["gamma2"])
        return self.params["gamma2"] * Pbar / e

    def max_cs2_up_to(self, Pc_max):
        """Exact max of cs^2 over every pressure occupied by a star with central pressure <= Pc_max."""
        if getattr(self, "is_hadronic", False) or Pc_max <= self.P_t:
            return float(self.table.cs2_max_below(min(Pc_max, self.table.Pbar_max)))
        g1, g2 = self.params["gamma1"], self.params["gamma2"]
        vals = [self.table.cs2_max_below(self.P_t)]
        # polytrope 1: cs^2 = g P/eps is monotone in P over the segment -> endpoints
        top1 = min(Pc_max, self.P_1)
        for P in (self.P_t, top1):
            e = (P / self.K1) ** (1 / g1)
            vals.append(g1 * P / e)
        if Pc_max > self.P_1:
            for P in (self.P_1, Pc_max):
                e = (P / self.K2) ** (1 / g2)
                vals.append(g2 * P / e)
        return float(max(vals))

    def seidov(self):
        """Seidov thresholds: unstable immediately after a transition if d/eps > 1/2 + 3P/(2 eps)."""
        out = []
        for (d, e, P) in ((self.d_t, self.eps_t, self.P_t), (self.d_1, self.eps_1b, self.P_1)):
            out.append(dict(ratio=d / e, threshold=0.5 + 1.5 * P / e, unstable=d / e > 0.5 + 1.5 * P / e))
        return out

    # ---------------- Maxwell bookkeeping: n(eps) and mu(P) ----------------
    def number_density(self, eps_c_max=3000.0, n_pts=600):
        """Reconstruct the baryon number density across the whole hybrid EOS.

        The polytropes are written in energy density, so they fix P(eps) but say
        nothing about n.  n is recovered from the first law at zero temperature,

            d(eps/n)/d(1/n) = -P   <=>   dn/n = d(eps)/(eps + P),

        integrated outward from the hadronic table's own n column.  Across each
        first-order transition the Maxwell conditions are that P and the baryon
        chemical potential mu = (eps + P)/n are both continuous, so the density on
        the far side of a jump is NOT free:

            n_+ = n_- (eps_- + Delta_eps + P) / (eps_- + P).

        Because Delta_eps >= 0 and P > 0 this gives Delta_n >= 0 automatically, so
        the construction is a genuine Maxwell coexistence rather than an energy
        jump imposed by hand -- which is what justifies calling it one in print.
        Returns a dict of arrays (MeV/fm^3, MeV/fm^3, fm^-3, MeV) plus the per-jump
        numbers (n_t, Delta_n, mu) a reader will ask for.
        """
        if getattr(self, "is_hadronic", False) or not self.valid:
            raise ValueError("number_density() needs a valid hybrid parameter vector")
        g1, g2 = self.params["gamma1"], self.params["gamma2"]
        eps_t = self.eps_t * P0                       # MeV/fm^3, as the integrator sees it
        P_t, P_1 = self.P_t * P0, self.P_1 * P0
        eps_1, d_t, d_1 = self.eps_1b * P0, self.d_t * P0, self.d_1 * P0
        K1 = self.K1 * P0 ** (1 - g1)                 # so that P[MeV/fm^3] = K1 eps^g1
        K2 = self.K2 * P0 ** (1 - g2)

        def integrate(e0, n0, e1, K, g, m):
            e = np.linspace(e0, e1, m)
            P = K * e ** g
            ln_n = n0 and np.log(n0)
            y = np.concatenate([[0.0], np.cumsum(np.diff(e) / (0.5 * ((e + P)[1:] + (e + P)[:-1])))])
            return e, P, np.exp(ln_n + y)

        # hadronic segment
        sel = self.table.eps <= eps_t
        e0, P0a, n0a = self.table.eps[sel], self.table.P[sel], self.table.n[sel]
        n_t = self.table.n_of_eps(eps_t)
        mu_t = (eps_t + P_t) / n_t
        # jump 1 (mu continuous)
        n_s1 = (eps_t + d_t + P_t) / mu_t
        e1, P1a, n1a = integrate(eps_t + d_t, n_s1, eps_1, K1, g1, n_pts)
        n_1 = n1a[-1]
        mu_1 = (eps_1 + P_1) / n_1
        # jump 2
        n_s2 = (eps_1 + d_1 + P_1) / mu_1
        e_top = max(eps_c_max, (eps_1 + d_1) * 1.2)
        e2, P2a, n2a = integrate(eps_1 + d_1, n_s2, e_top, K2, g2, n_pts)

        eps = np.concatenate([e0, e1, e2])
        P = np.concatenate([P0a, P1a, P2a])
        n = np.concatenate([n0a, n1a, n2a])
        return dict(eps=eps, P=P, n=n, mu=(eps + P) / n,
                    jumps=[dict(eps=eps_t, P=P_t, mu=mu_t, n_lo=n_t, n_hi=n_s1,
                                d_eps=d_t, d_n=n_s1 - n_t),
                           dict(eps=eps_1, P=P_1, mu=mu_1, n_lo=n_1, n_hi=n_s2,
                                d_eps=d_1, d_n=n_s2 - n_1)])

    def maxwell_report(self, n0=0.16, n_floor=1e-3):
        """Human-readable check that the Maxwell construction is self-consistent.

        Checked only above `n_floor`.  Tabulated outer crusts are routinely
        stitched from sources whose (eps, P, n) triples are not exactly
        thermodynamically consistent, so mu wanders by a few MeV around 930 MeV
        below n ~ 1e-11 fm^-3.  Those layers carry no mass, radius or tidal
        signal; the separate `crust_mu_steps` count records the issue rather than
        hiding it.
        """
        d = self.number_density()
        m = d["n"] > n_floor
        ok_n = bool(np.all(np.diff(d["n"][m]) >= -1e-12))
        ok_mu = bool(np.all(np.diff(d["mu"][m]) >= -1e-9 * np.maximum(d["mu"][m][:-1], 1)))
        dmu = np.diff(d["mu"][~m])
        n_crust_bad = int(np.sum(dmu < 0))
        worst = float(dmu.min()) if dmu.size else 0.0
        lines = [f"n monotonically increasing: {ok_n};  mu monotonically increasing: {ok_mu}"
                 f"   (above n = {n_floor:g} fm^-3)",
                 f"  tabulated crust below that floor: {n_crust_bad} non-monotone mu steps, "
                 f"worst {worst:+.3f} MeV out of ~938 -- an inconsistency in the supplied "
                 f"(eps, P, n) table, not in the construction, and massless for M/R/Lambda"]
        for i, j in enumerate(d["jumps"], 1):
            lines.append(f"  jump {i}: n = {j['n_lo']/n0:.2f} -> {j['n_hi']/n0:.2f} n0  "
                         f"(Delta n/n = {j['d_n']/j['n_lo']:.3f}),  mu = {j['mu']:.1f} MeV,  "
                         f"Delta eps/eps = {j['d_eps']/j['eps']:.3f},  P = {j['P']:.2f} MeV/fm^3")
        return ok_n and ok_mu, "\n".join(lines)
