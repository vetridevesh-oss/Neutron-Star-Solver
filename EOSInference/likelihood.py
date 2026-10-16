"""
likelihood.py -- astrophysical likelihood of a computed stellar sequence.

Every term has the same shape: a published posterior, divided by its sampling
prior to recover a likelihood, integrated against the EOS-conditioned population
prior over the star's mass (or, for the binary, over its mass ratio).

  L_i(theta) = int dm pi(m|theta) * K_i(m, R(m; theta))                [NICER]
  L_GW(theta) = int dq pi(q|theta) * K_GW(q, Lambda1(q;theta), Lambda2(q;theta))
  L_floor(theta) = Phi((M_TOV - mu)/sigma)                             [J0952]

pi(m|theta) and pi(q|theta) are *normalised over the EOS's own stable support*.
That normalisation is not cosmetic: it is the Occam factor that penalises an EOS
whose maximum mass is far above anything observed, and it is what makes the
branch rules below probability statements rather than ad-hoc choices.

Branch rules (config.LIKELIHOOD["branch_rule"]) are different joint priors over
(mass, branch):

  "mixture" : pick m uniformly on the union of the stable branches' mass support,
              then pick uniformly among the branches that exist at that m.
              Weight 1/N_b(m); normalised by |U| = measure of the union.  DEFAULT.
  "sum"     : pick (m, branch) uniformly on the disjoint union.  Weight 1;
              normalised by S = sum_b (mass span of branch b).  This is NOT
              double counting -- the larger normaliser pays for it -- but it is a
              different population prior, and it adds an extra Occam penalty on
              EOSs with a lot of stable mass range.
  "max"     : take the best branch.  This is a profile, not a marginal, so it is
              not a likelihood; it is here only to compare against Grundler & Li.

GW170817 details
----------------
The chirp mass is measured to 6e-5 relative, so the binary is parameterised by
(q, Lambda1, Lambda2) at fixed Mc.  The released posterior's sampling prior is
flat in Lambda1 and Lambda2 on [0, 5000] (verified from the prior samples) but
*not* flat in q: uniform in component masses becomes

    pi_s(q) propto q^(-6/5) (1+q)^(2/5)      at fixed Mc

after the change of variables.  Taking the EOS-side population prior to be
uniform in component masses as well, the two q factors cancel inside the
integral and survive only in the normaliser:

    L_GW(theta) = (1/Z(theta)) int_{A(theta)} dq  p_post(q, Lambda1, Lambda2),
    Z(theta)    = int_{A(theta)} dq  pi_s(q),

with A(theta) the mass ratios for which both components are on a stable branch
of this EOS (in particular m1 <= M_TOV).  Set LIKELIHOOD["gw_population"] to
"uniform_mass" for that, or to "flat_q" to use a flat q population prior instead,
in which case pi_s(q) is divided out explicitly.
"""
import numpy as np
from scipy.special import log_ndtr, ndtr

import config
import datasets as D

_trapz = getattr(np, "trapezoid", None) or np.trapz
LOG_FLOOR = -690.0          # ln(1e-300): a term that vanishes, without -inf


# ---------------------------------------------------------------------------
# branch combination
# ---------------------------------------------------------------------------
def _combine(dens, valid, rule):
    """Combine per-branch densities over axis 0; returns (integrand, n_branches)."""
    dens = np.where(valid, dens, 0.0)
    n = valid.sum(0)
    if rule == "sum":
        return dens.sum(0), n
    if rule == "max":
        return dens.max(0), n
    return np.where(n > 0, dens.sum(0) / np.maximum(n, 1), 0.0), n


def _cosine(lo, hi, n):
    """n points on [lo, hi], clustered quadratically towards both ends."""
    k = np.arange(n)
    return 0.5 * (lo + hi) - 0.5 * (hi - lo) * np.cos(np.pi * k / (n - 1))


def quad_grid(lo, hi, segments, n, focus=None):
    """Quadrature abscissae for an integrand with square-root endpoint behaviour.

    R(m) and Lambda(m) go like sqrt(m_max - m) near the end of a stable branch
    (dM/dP_c vanishes there), so their derivative is infinite at the endpoint and
    a uniform trapezoid rule converges like n^(-3/2).  On J0740 that left a 0.08
    nat error at n = 150 and the value was still drifting at n = 2400.  Clustering
    points quadratically at the ends of the full range AND at each branch's own
    endpoints removes it: the same 150 points then give the n -> infinity answer
    to about 0.001 nat.  `focus` adds a third cluster over the interval where the
    data have support, which is where the integrand is nonzero at all and where
    the kernel is narrowest.
    """
    g = [_cosine(lo, hi, n)]
    if focus is not None:                      # where the data actually have support
        a, b = max(focus[0], lo), min(focus[1], hi)
        if b > a:
            g.append(_cosine(a, b, n))
    for a, b in segments:
        a, b = max(a, lo), min(b, hi)
        if b > a:
            g.append(_cosine(a, b, max(8, n // 3)))
    return np.unique(np.concatenate(g))


def _normaliser(n_branch, grid, rule):
    """Measure of the population prior's support, on the same quadrature grid.

    mixture / max : |U|, the measure of the union of the branches' support.
    sum           : S, the measure of the disjoint union (branches counted once each).
    """
    if rule == "sum":
        return float(_trapz(n_branch.astype(float), grid))
    return float(_trapz((n_branch > 0).astype(float), grid))


# ---------------------------------------------------------------------------
# terms
# ---------------------------------------------------------------------------
class MassRadiusTerm:
    """One NICER-style (M, R) source."""

    def __init__(self, name, samples, kde, m_min, n_grid):
        self.name, self.samples, self.kde = name, samples, kde
        self.m_min, self.n_grid = m_min, n_grid
        box_lo, box_hi = kde.box()
        self.m_lo_data, self.m_hi_data = float(box_lo[0]), float(box_hi[0])

    def __call__(self, seq):
        m_lo = max(self.m_min, self.m_lo_data)
        m_hi = min(seq.M_tov, self.m_hi_data)
        if m_hi <= m_lo:
            return 0.0
        # quadrature over the full prior support, so that the normaliser and the
        # integral are computed on the same grid; the kernel is zero where the
        # data have no support, so extending the grid costs nothing but is what
        # makes 1/|U| the correct Occam factor
        m = quad_grid(self.m_min, seq.M_tov,
                      [(b["M"][0], b["M"][-1]) for b in seq.branches], self.n_grid,
                      focus=(self.m_lo_data, self.m_hi_data))
        R = seq.branch_values(m, "R")                            # (n_branch, n_m)
        valid = np.isfinite(R)
        pts = np.stack([np.broadcast_to(m, R.shape), np.where(valid, R, np.nan)], -1)
        dens = self.kde(pts)
        integrand, nb = _combine(dens, valid, self.rule)
        norm = _normaliser(nb, m, self.rule)
        if norm <= 0:
            return 0.0
        return float(_trapz(integrand, m) / norm)


def _q_of_m1(Mc, m1_target, q_min=0.05):
    """Invert m1(q) at fixed chirp mass; m1 decreases monotonically in q."""
    lo, hi = q_min, 1.0
    if D.masses_from_Mc_q(Mc, lo)[0] < m1_target or D.masses_from_Mc_q(Mc, hi)[0] > m1_target:
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if D.masses_from_Mc_q(Mc, mid)[0] > m1_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class GWTerm:
    """GW170817 at fixed chirp mass, in (q, Lambda1, Lambda2)."""

    def __init__(self, name, samples, kde, Mc, m_min, n_grid, population="uniform_mass",
                 q_min=0.2):
        self.name, self.samples, self.kde = name, samples, kde
        self.Mc, self.m_min, self.n_grid = Mc, m_min, n_grid
        self.population, self.q_min = population, q_min
        if population not in ("uniform_mass", "flat_q"):
            raise ValueError("gw_population must be 'uniform_mass' or 'flat_q'")

    def q_lo(self, M_tov):
        """Smallest q with m1 <= M_tov and m2 >= m_min.

        m1 falls and m2 rises with q, so both constraints are one-sided and the
        bound is found by bisection.  Gridding from here rather than from a fixed
        q_min puts every quadrature point inside the support and resolves the
        edge exactly, which matters when M_TOV is close to the equal-mass m1.
        """
        lo, hi = self.q_min, 1.0
        f = lambda q: min(M_tov - D.masses_from_Mc_q(self.Mc, q)[0],
                          D.masses_from_Mc_q(self.Mc, q)[1] - self.m_min)
        if f(hi) < 0:
            return None                      # no allowed mass ratio at all
        if f(lo) >= 0:
            return lo
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if f(mid) >= 0:
                hi = mid
            else:
                lo = mid
        return hi

    def __call__(self, seq):
        qlo = self.q_lo(seq.M_tov)
        if qlo is None or qlo >= 1.0:
            return 0.0
        # same endpoint behaviour in Lambda(m1): cluster at the ends of the
        # allowed range and at the mass ratios where a branch starts or stops
        segs = []
        for b in seq.branches:
            for mm in (b["M"][0], b["M"][-1]):
                r = _q_of_m1(self.Mc, mm)
                if r is not None and qlo < r < 1.0:
                    segs.append((max(qlo, r - 0.05), min(1.0, r + 0.05)))
        q = quad_grid(qlo, 1.0, segs, self.n_grid)
        m1, m2 = D.masses_from_Mc_q(self.Mc, q)
        ok = (m1 <= seq.M_tov + 1e-12) & (m2 >= self.m_min - 1e-12)
        if not ok.any():
            return 0.0
        L1 = seq.branch_values(m1, "Lam")                        # (nb, nq)
        L2 = seq.branch_values(m2, "Lam")
        v1, v2 = np.isfinite(L1) & ok, np.isfinite(L2) & ok
        nb = L1.shape[0]
        dens = np.zeros((nb * nb, len(q)))
        valid = np.zeros((nb * nb, len(q)), bool)
        qb = np.broadcast_to(q, L1.shape[1:])
        for i in range(nb):
            for j in range(nb):
                k = i * nb + j
                valid[k] = v1[i] & v2[j]
                pts = np.stack([qb, np.where(v1[i], L1[i], np.nan),
                                np.where(v2[j], L2[j], np.nan)], -1)
                dens[k] = self.kde(pts)
        integrand, n_pair = _combine(dens, valid, self.rule)
        support = n_pair > 0
        if not support.any():
            return 0.0
        pi_s = D.prior_divide({"kind": "q_from_uniform_m"}, q)
        if self.population == "uniform_mass":
            # pi(q|theta) propto pi_s(q) on the support: the two q-shapes cancel
            # inside the integral and survive only in the normaliser
            w_int = np.ones_like(q)
            measure = pi_s
        else:
            # flat population prior in q: divide the sampling prior out explicitly
            w_int = np.where(pi_s > 0, 1.0 / pi_s, 0.0)
            measure = np.ones_like(q)
        # "sum" integrates over the disjoint union of (branch1, branch2) cells, so
        # its normaliser counts each cell; the others normalise over the union
        count = n_pair.astype(float) if self.rule == "sum" else (n_pair > 0).astype(float)
        norm = float(_trapz(measure * count, q))
        if norm <= 0:
            return 0.0
        return float(_trapz(np.where(support, integrand * w_int, 0.0), q) / norm)


class MassFloorTerm:
    """A pulsar with a mass measurement and no radius (PSR J0952-0607)."""

    def __init__(self, cfg, m_min, mode="cdf"):
        self.mean, self.sigma = cfg["mean"], cfg["sigma"]
        self.m_min, self.mode = m_min, mode
        if mode not in ("cdf", "marginal"):
            raise ValueError("mass_floor mode must be 'cdf' or 'marginal'")

    def logL(self, seq):
        z_hi = (seq.M_tov - self.mean) / self.sigma
        if self.mode == "cdf":
            # P(m_measured < M_TOV): the conventional "mass floor".  Equivalent to
            # an improper flat prior on m, so it carries no 1/|U| Occam factor and
            # is therefore NOT on the same footing as the NICER terms.
            return float(log_ndtr(z_hi))
        z_lo = (self.m_min - self.mean) / self.sigma
        p = ndtr(z_hi) - ndtr(z_lo)
        span = max(seq.M_tov - self.m_min, 1e-12)
        return float(np.log(max(p / span, 1e-300)))


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
class Likelihood:
    """All observational terms.  Construct once and share across dynesty workers."""

    def __init__(self, mr_terms=None, gw_term=None, mass_floor=None, like_cfg=None,
                 data_cfg=None, verbose=False):
        lc = dict(like_cfg or config.LIKELIHOOD)
        dc = data_cfg or config.DATA
        self.lc = lc
        self.rule = lc["branch_rule"]
        if self.rule not in ("mixture", "max", "sum"):
            raise ValueError("branch_rule must be mixture / max / sum")
        self.m_min = lc["mass_prior_min"]
        self.provenance = []

        if mr_terms is None:
            mr_terms = []
            for name, spec in dc.get("mr_sources", {}).items():
                if not spec.get("use", True):
                    continue
                ss = D.load_text(spec, name)
                self.provenance.append(self._prov(ss, spec))
                for axis, key in ((0, "M"), (1, "R")):
                    if spec.get("headline", {}).get(key):
                        msg = D.check_headline(ss, axis, spec["headline"][key], key)
                        if verbose:
                            print("  [check]", msg)
                kde = D.GridKDE(ss.x, ss.w, bw_factor=spec.get("bw_factor", lc["bw_factor_mr"]),
                                bounds=spec.get("bounds", ()))
                mr_terms.append(MassRadiusTerm(name, ss, kde, self.m_min, lc["n_mass_grid"]))
        self.mr_terms = mr_terms

        if gw_term is None and dc.get("gw") and dc["gw"].get("use", True):
            spec = dc["gw"]
            ss = D.load_gw_hdf5(spec)
            self.provenance.append(self._prov(ss, spec))
            kde = D.GridKDE(ss.x, ss.w, bw_factor=spec.get("bw_factor", lc["bw_factor_gw"]),
                            bounds=spec.get("bounds", ()))
            gw_term = GWTerm(ss.name, ss, kde, ss.Mc, self.m_min, lc["n_q_grid"],
                             lc["gw_population"], lc["q_min"])
        self.gw_term = gw_term

        mf = mass_floor if mass_floor is not None else dc.get("mass_floor")
        self.floor = MassFloorTerm(mf, self.m_min, lc["mass_floor_mode"]) if mf else None

        for t in self.mr_terms:
            t.rule = self.rule
        if self.gw_term is not None:
            self.gw_term.rule = self.rule

    @staticmethod
    def _prov(ss, spec):
        return dict(name=ss.name, file=ss.path, doi=spec.get("doi", ""), n=len(ss.x),
                    ess=float(ss.ess), cols=ss.cols, note=spec.get("note", ""))

    def components(self, seq):
        out = {}
        for t in self.mr_terms:
            out[t.name] = float(np.log(max(t(seq), 1e-300)))
        if self.gw_term is not None:
            out[self.gw_term.name] = float(np.log(max(self.gw_term(seq), 1e-300)))
        if self.floor is not None:
            out["mass_floor"] = self.floor.logL(seq)
        return out

    def __call__(self, seq):
        # each component is already floored at LOG_FLOOR, so the sum is finite;
        # dynesty copes with -inf but a finite floor keeps its bounds better behaved
        return float(sum(self.components(seq).values()))

    def describe(self):
        lines = [f"likelihood: branch_rule={self.rule}  m_min={self.m_min}  "
                 f"gw_population={self.lc['gw_population']}  "
                 f"mass_floor_mode={self.lc['mass_floor_mode']}"]
        for p in self.provenance:
            lines.append(f"  {p['name']:<10s} N={p['n']:>8d} ESS={p['ess']:>9.0f}  "
                         f"cols={p['cols']}  {p['doi']}")
        return "\n".join(lines)
