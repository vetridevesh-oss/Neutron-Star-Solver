"""
datasets.py -- loading published posterior samples and turning them into densities.

Two jobs, kept away from the physics:

1. **Loading with provenance.**  Every file is declared in `config.DATA` with its
   own column mapping, DOI and md5.  Column *order differs between releases*
   (J0740 and J0030 are (weight, M, R); the Miller J0437 file is (R, M, weight))
   and a silent transposition is the single easiest way to get a wrong answer
   here, so the loader records what it read and `check_headline` re-derives the
   published median and 68% interval from the loaded arrays.

2. **Density estimation.**  A released posterior is  p(d|x) pi_s(x) / Z.  What the
   EOS likelihood needs is the first factor, so the sampling prior pi_s has to be
   divided out (`prior_divide` below).  Densities are weighted Gaussian KDEs
   evaluated by FFT on a regular grid in whitened coordinates:

     * weights are kept (no resampling down to a few thousand points -- that
       throws away most of the information in a 1e6-sample file and widens the
       bandwidth by n^(-1/(d+4))),
     * the bandwidth is `bw_factor` x Silverman computed from the *effective*
       sample size, and `bw_factor` is a per-source setting because the files
       differ in size by two orders of magnitude,
     * hard boundaries of the sampling prior (R <= 16 km, Lambda >= 0, q <= 1)
       are handled by mirroring the samples that lie within a few bandwidths of
       the edge, which imposes the reflecting condition the truncation implies
       instead of letting the kernel leak probability out of the support.
"""
import hashlib
import os

import numpy as np
from scipy.signal import fftconvolve

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class SampleSet:
    """Weighted posterior samples plus where they came from."""

    def __init__(self, x, w, name, path, cols, doi="", note=""):
        x = np.atleast_2d(np.asarray(x, float))
        w = np.ones(len(x)) if w is None else np.asarray(w, float)
        keep = np.isfinite(w) & (w > 0) & np.all(np.isfinite(x), axis=1)
        self.x, self.w = x[keep], w[keep] / w[keep].sum()
        self.name, self.path, self.cols, self.doi, self.note = name, path, cols, doi, note
        self.n_raw, self.n_dropped = len(x), int((~keep).sum())

    @property
    def ess(self):
        return 1.0 / np.sum(self.w ** 2)

    def quantile(self, j, q):
        o = np.argsort(self.x[:, j])
        c = np.cumsum(self.w[o])
        return np.interp(q, c / c[-1], self.x[o, j])

    def summary(self, j):
        lo, med, hi = (self.quantile(j, p) for p in (0.16, 0.5, 0.84))
        return med, med - lo, hi - med

    def __repr__(self):
        return (f"<SampleSet {self.name}: N={len(self.x)} ESS={self.ess:.0f} "
                f"cols={self.cols} from {os.path.basename(self.path)}>")


def load_text(spec, name):
    """Load a plain-text sample file described by a config.DATA entry."""
    path = spec["file"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name}: {path} (set config.DATA['{name}']['file'])")
    if spec.get("md5") and md5(path) != spec["md5"]:
        raise ValueError(f"{name}: md5 of {path} does not match the manifest in config.DATA; "
                         "the file is not the release this pipeline was validated against")
    arr = np.loadtxt(path, comments="#")
    c = spec["cols"]                                   # dict: logical name -> column index
    x = np.column_stack([arr[:, c[k]] for k in spec["order"]])
    w = arr[:, c["weight"]] if "weight" in c and c["weight"] is not None else None
    return SampleSet(x, w, name, path, c, spec.get("doi", ""), spec.get("note", ""))


def check_headline(ss, j, expected, label, tol_med=0.02, tol_err=0.15):
    """Re-derive a published median and 68% interval; raise if they disagree.

    `expected` is (median, minus, plus).  This is the guard against a column
    transposition or a mis-set weight column: both reproduce a plausible-looking
    density but not the published number.
    """
    med, lo, hi = ss.summary(j)
    e_med, e_lo, e_hi = expected
    bad = (abs(med - e_med) > tol_med
           or abs(lo - e_lo) > max(tol_err * e_lo, 0.02)
           or abs(hi - e_hi) > max(tol_err * e_hi, 0.02))
    msg = (f"{ss.name} {label}: {med:.3f} -{lo:.3f} +{hi:.3f}   "
           f"(published {e_med:.3f} -{e_lo:.3f} +{e_hi:.3f})")
    if bad:
        raise ValueError("HEADLINE MISMATCH -- check columns/weights in config.DATA.\n  " + msg)
    return msg


# ---------------------------------------------------------------------------
# weighted KDE on a grid
# ---------------------------------------------------------------------------
def silverman(n_eff, d):
    """Silverman's rule-of-thumb bandwidth in units of the (whitened) standard deviation."""
    return (n_eff * (d + 2) / 4.0) ** (-1.0 / (d + 4))


class GridKDE:
    """Weighted Gaussian KDE in d dimensions, tabulated by FFT on a regular grid.

    The samples are whitened with their weighted covariance, so the kernel is the
    isotropic Gaussian of width `h` = bw_factor * Silverman(ESS, d) and the grid
    is axis-aligned in whitened space.  Evaluation is multilinear interpolation;
    the density is zero outside the tabulated box (which extends `pad` bandwidths
    beyond the samples, so the truncation is in the far tail).

    `bounds` declares hard edges of the SAMPLING PRIOR in the original
    coordinates as a list of (axis, value, side) with side in {"lo", "hi"};
    samples within `reflect_h` bandwidths of such an edge are mirrored through it.
    """

    def __init__(self, x, w, bw_factor=1.0, n_grid=None, pad=4.0, bounds=(),
                 reflect_h=4.0, max_cells=4_000_000, pts_per_h=3.0):
        x = np.asarray(x, float)
        w = np.asarray(w, float)
        w = w / w.sum()
        self.d = x.shape[1]
        self.n_eff = 1.0 / np.sum(w ** 2)
        self.h = float(bw_factor * silverman(self.n_eff, self.d))
        self.bw_factor = bw_factor

        # whitening from the weighted covariance
        self.mean = w @ x
        dx = x - self.mean
        cov = (dx * w[:, None]).T @ dx / (1.0 - np.sum(w ** 2))
        self.L = np.linalg.cholesky(cov)                      # cov = L L^T
        self.Linv = np.linalg.inv(self.L)
        self.sd = np.sqrt(np.diag(cov))

        # reflect across hard prior edges (in the ORIGINAL coordinates)
        xs, ws = [x], [w]
        for (ax, val, side) in bounds:
            near = (np.abs(x[:, ax] - val) < reflect_h * self.h * self.sd[ax])
            if side == "hi":
                near &= x[:, ax] <= val
            else:
                near &= x[:, ax] >= val
            if near.any():
                m = x[near].copy()
                m[:, ax] = 2 * val - m[:, ax]
                xs.append(m)
                ws.append(w[near])
        x = np.vstack(xs)
        w = np.concatenate(ws)
        self.bounds = tuple(bounds)

        u = (x - self.mean) @ self.Linv.T                     # whitened samples
        lo = u.min(0) - pad * self.h
        hi = u.max(0) + pad * self.h
        if n_grid is None:
            n = np.maximum(32, np.ceil(pts_per_h * (hi - lo) / self.h)).astype(int)
            while np.prod(n.astype(float)) > max_cells:
                n = np.maximum(32, (n * 0.9).astype(int))
            n_grid = n
        self.n_grid = np.atleast_1d(np.asarray(n_grid, int)) * np.ones(self.d, int)
        self.lo, self.hi = lo, hi
        self.step = (hi - lo) / (self.n_grid - 1)

        # linear binning of the weights, then convolution with the Gaussian kernel
        edges = [np.linspace(lo[i] - 0.5 * self.step[i], hi[i] + 0.5 * self.step[i],
                             self.n_grid[i] + 1) for i in range(self.d)]
        H, _ = np.histogramdd(u, bins=edges, weights=w)
        ker = 1.0
        for i in range(self.d):
            m = int(np.ceil(4.0 * self.h / self.step[i]))
            t = np.arange(-m, m + 1) * self.step[i]
            k = np.exp(-0.5 * (t / self.h) ** 2)
            k /= k.sum()
            ker = np.multiply.outer(ker, k) if i else k
        dens = fftconvolve(H, ker, mode="same")
        dens = np.maximum(dens, 0.0)
        # normalise as a density in WHITENED coordinates; the Jacobian det(L) is
        # applied on evaluation so that the result is a density in x
        vol = np.prod(self.step)
        tot = dens.sum() * vol
        self.dens = dens / max(tot, 1e-300)
        self.jac = 1.0 / abs(np.linalg.det(self.L))

    def __call__(self, pts):
        """pts: (..., d) in the original coordinates -> density, 0 outside the box."""
        pts = np.asarray(pts, float)
        shape = pts.shape[:-1]
        q = pts.reshape(-1, self.d)
        ok = np.all(np.isfinite(q), axis=1)
        u = np.zeros_like(q)
        u[ok] = (q[ok] - self.mean) @ self.Linv.T
        idx = (u - self.lo) / self.step
        out = np.zeros(len(q))
        inside = ok & np.all((idx >= 0) & (idx <= self.n_grid - 1), axis=1)
        if inside.any():
            from scipy.ndimage import map_coordinates
            out[inside] = map_coordinates(self.dens, idx[inside].T, order=1,
                                          mode="constant", cval=0.0)
        return (out * self.jac).reshape(shape)

    def box(self, pad=0.0):
        """Bounding box of the tabulated density in the ORIGINAL coordinates."""
        corners = np.array(np.meshgrid(*[[self.lo[i], self.hi[i]] for i in range(self.d)],
                                       indexing="ij")).reshape(self.d, -1).T
        x = corners @ self.L.T + self.mean
        return x.min(0) - pad, x.max(0) + pad

    def marginal_quantile(self, axis, qs, n=300):
        """Quantiles of the tabulated density along one ORIGINAL axis.

        Validation only: integrates the density on a regular grid in the original
        coordinates, so it is directly comparable with the sample quantiles.
        """
        lo, hi = self.box()
        g = [np.linspace(lo[i], hi[i], n) for i in range(self.d)]
        mesh = np.meshgrid(*g, indexing="ij")
        v = self(np.stack([m for m in mesh], -1))
        prof = v.sum(axis=tuple(i for i in range(self.d) if i != axis))
        c = np.cumsum(prof)
        return np.interp(qs, c / c[-1], g[axis])


# ---------------------------------------------------------------------------
# sampling-prior division
# ---------------------------------------------------------------------------
def prior_divide(spec, pts):
    """Sampling-prior density pi_s evaluated at `pts`, for the axes declared in `spec`.

    Anything uniform contributes a constant, which cancels in every Bayes factor,
    so only *non-uniform* sampling priors need to appear here.  Supported kinds:

      "uniform"        : constant (returns 1)
      "q_from_uniform_m": pi(q) ∝ q^(-6/5) (1+q)^(2/5) at fixed chirp mass,
                          which is what "uniform in component masses" becomes
                          after the change of variables (m1, m2) -> (Mc, q).
    """
    kind = spec.get("kind", "uniform")
    if kind == "uniform":
        return np.ones(np.shape(pts)[:-1] if np.ndim(pts) > 1 else np.shape(pts))
    if kind == "q_from_uniform_m":
        q = np.asarray(pts, float)
        return np.where(q > 0, q ** (-1.2) * (1.0 + q) ** 0.4, 0.0)
    raise ValueError(f"unknown sampling-prior kind {kind!r}")


# ---------------------------------------------------------------------------
# GW170817
# ---------------------------------------------------------------------------
def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def masses_from_Mc_q(Mc, q):
    """(m1, m2) from the chirp mass and the mass ratio q = m2/m1 <= 1."""
    m1 = Mc * (1.0 + q) ** 0.2 / q ** 0.6
    return m1, q * m1


def load_gw_hdf5(spec, name="GW170817"):
    """Load a GWTC-style HDF5 posterior and convert the masses to the source frame.

    Returns a SampleSet with columns (q, Lambda1, Lambda2) and the source-frame
    chirp mass stored on `.Mc`.

    The released masses are DETECTOR-frame; the source frame needs a redshift.
    `spec["redshift"]` chooses how:
      {"mode": "fixed", "z": ...}       -- a single redshift, normally the one from
                                           the electromagnetic counterpart's host.
                                           This keeps Mc_src as sharply determined
                                           as Mc_det (rel. width 6e-5).
      {"mode": "per_sample", "H0":, "Om":} -- z from each sample's luminosity
                                           distance.  Mc_src then inherits the
                                           distance uncertainty (rel. width 2e-3),
                                           which is why the fixed-z form is the
                                           default; the difference is a sensitivity
                                           test, not a correction.
    """
    import h5py
    path = spec["file"]
    if spec.get("md5") and md5(path) != spec["md5"]:
        raise ValueError(f"{name}: md5 of {path} does not match the manifest in config.DATA")
    with h5py.File(path, "r") as f:
        d = f[spec["dataset"]][:]
    m1d = d["m1_detector_frame_Msun"].astype(float)
    m2d = d["m2_detector_frame_Msun"].astype(float)
    rs = spec["redshift"]
    if rs["mode"] == "fixed":
        z = float(rs["z"]) * np.ones(len(d))
    elif rs["mode"] == "per_sample":
        c_km = 299792.458
        z = rs["H0"] * d["luminosity_distance_Mpc"].astype(float) / c_km
    else:
        raise ValueError("redshift mode must be 'fixed' or 'per_sample'")
    m1, m2 = m1d / (1 + z), m2d / (1 + z)
    q = m2 / m1
    x = np.column_stack([q, d["lambda1"].astype(float), d["lambda2"].astype(float)])
    ss = SampleSet(x, None, name, path, {"q": 0, "Lambda1": 1, "Lambda2": 2},
                   spec.get("doi", ""), spec.get("note", ""))
    ss.Mc = float(np.median(chirp_mass(m1, m2)[np.all(np.isfinite(x), axis=1)]))
    ss.Mc_rel_width = float(np.std(chirp_mass(m1, m2)) / ss.Mc)
    ss.z = float(np.median(z))
    return ss
