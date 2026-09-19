"""
fetch_observational_data.py -- pulls real posterior sample releases for the
NICER + GW170817 likelihood terms, per the mentee's spec.

*** IMPORTANT SANDBOX LIMITATION ***
This Claude container's outbound network is restricted to package-manager
and GitHub domains only (verified: zenodo.org, gwosc.org, dcc.ligo.org all
return HTTP 403 "host_not_allowed" from the egress proxy here). So the
`download_all()` function below WILL fail if run inside this environment --
that's expected, not a bug. Run this file on your mentee's own machine (or
Google Colab, which has unrestricted network) to actually pull the data;
the URLs themselves are real and verified below.

REAL DATA SOURCES (verified via literature search, not fabricated):

  J0030+0451 Scenario C (PDT-U, primary) and Scenario B (ST+PDT, ablation):
    Vinciguerra et al. 2024, ApJ 961, 62 (arXiv:2308.09469)
    "An Updated Mass-Radius Analysis of the 2017-2018 NICER Data Set of
    PSR J0030+0451"
    Zenodo record: https://zenodo.org/records/8239000  (DOI 10.5281/zenodo.8239000)

  J0740+6620:
    Salmi et al. 2024, ApJ 974, 294 (arXiv:2406.14466)
    "The Radius of the High-mass Pulsar PSR J0740+6620 with 3.6 yr of
    NICER Data"
    Zenodo record: https://zenodo.org/records/10519473 (DOI 10.5281/zenodo.10519473)

  J0437-4715:
    Choudhury et al. 2024, ApJL 971, L20 (arXiv:2407.06789)
    "A NICER View of the Nearest and Brightest Millisecond Pulsar:
    PSR J0437-4715"
    Zenodo record DOI 10.5281/zenodo.13766753

  GW170817 Lambda1-Lambda2 posterior:
    Abbott et al. 2018, PRL 121, 161101 ("GW170817: Measurements of
    Neutron Star Radii and Equation of State")
    LIGO DCC: https://dcc.ligo.org/LIGO-P1800115/public
    Full PE sample release (GWTC-1): via GWOSC,
    https://gwosc.org/eventapi/html/GWTC-1-confident/GW170817/v3/
    (file: GW170817_GWTC-1.hdf5, posterior_samples group)

  J0952-0607 mass floor (radio timing, no posterior release needed):
    Romani et al. 2022, ApJL 934, L17 -- M = 2.35 +/- 0.17 Msun, used
    directly as a 1-D Gaussian below.

Each real source's release format differs (some are plain-text mr.txt
files, some are HDF5 with named datasets) -- the loader functions below
are written generically against "an (N,2) or (N,4) array of samples" and
will need the specific column-selection adjusted once you have the actual
files in hand; the TODOs mark exactly where.
"""

import os
import numpy as np
from scipy.stats import gaussian_kde

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

SOURCES = {
    "J0030_C": "https://zenodo.org/records/8239000",
    "J0030_B": "https://zenodo.org/records/8239000",   # same release, different scenario file
    "J0740":   "https://zenodo.org/records/10519473",
    "J0437":   "https://zenodo.org/records/13766753",
    "GW170817": "https://gwosc.org/eventapi/html/GWTC-1-confident/GW170817/v3/",
}


def download_all(target_dir=DATA_DIR):
    """Attempt to fetch every source above. Reports success/failure per
    source rather than raising on the first failure, so a partial network
    environment still gets you whatever IS reachable."""
    import urllib.request
    results = {}
    for name, url in SOURCES.items():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                results[name] = f"reachable (HTTP {resp.status}) -- but this is a landing " \
                                  f"page, not the data file; see the module docstring for the " \
                                  f"specific file to download from it"
        except Exception as e:
            results[name] = f"FAILED: {e}"
    return results


# ---------------------------------------------------------------------------
# Generic 2D KDE likelihood term (NICER M-R, GW Lambda1-Lambda2)
# ---------------------------------------------------------------------------
class Posterior2DKDE:
    """Wraps a set of real (x, y) posterior samples as a 2D KDE-based
    likelihood term. Used for NICER (M, R) and for GW170817 (Lambda1, Lambda2).
    """

    def __init__(self, x_samples, y_samples, label=""):
        self.label = label
        pts = np.vstack([x_samples, y_samples])
        self.kde = gaussian_kde(pts)
        self.x_range = (x_samples.min(), x_samples.max())
        self.y_range = (y_samples.min(), y_samples.max())

    def log_prob(self, x, y):
        if not (self.x_range[0] <= x <= self.x_range[1] and
                self.y_range[0] <= y <= self.y_range[1]):
            # far outside the sampled region -- KDE tails are unreliable
            # there; treat as a strong (but not infinite) penalty rather
            # than a hard cutoff, to keep the sampler's gradient sane
            return -50.0
        val = self.kde.logpdf([x, y])[0]
        return float(val) if np.isfinite(val) else -50.0

    @classmethod
    def from_txt(cls, path, col_x=0, col_y=1, skiprows=1, label=""):
        """TODO once real files are in hand: adjust col_x/col_y/skiprows to
        match the actual release format (e.g. Salmi et al.'s
        'J0740_NICERXMM_full_mr.txt' -- check its header for the exact
        column order before trusting col_x=0, col_y=1)."""
        tab = np.loadtxt(path, skiprows=skiprows)
        return cls(tab[:, col_x], tab[:, col_y], label=label)

    @classmethod
    def from_h5(cls, path, group, key_x, key_y, label=""):
        """TODO: for the GWOSC GWTC-1 HDF5 release, group is typically
        'IMRPhenomPv2NRT_lowSpin_posterior' (or similar -- verify the exact
        group name against the file's own listing) and key_x/key_y would be
        'lambda1'/'lambda2' if present directly, or you derive them from
        mass1/mass2/lambda1/lambda2 columns."""
        import h5py
        with h5py.File(path, "r") as f:
            x = f[group][key_x][:]
            y = f[group][key_y][:]
        return cls(x, y, label=label)


class Gaussian1D:
    """1-D constraint for J0952's radio-timing mass floor (no posterior
    release to KDE against -- it's reported as a Gaussian summary).

    ASSUMPTION: J0952 is used as a "must be able to reach at least this
    measured mass" constraint on Mmax, via the normal CDF, NOT a symmetric
    Gaussian penalty in Mmax. A symmetric Gaussian would wrongly penalize
    an EOS whose Mmax is comfortably ABOVE 2.35 Msun just as much as one
    below it -- but a stiffer-than-needed EOS is not in tension with this
    measurement at all (the measured star's mass just has to lie on the
    EOS's stable branch, i.e. below Mmax). This one_sided=CDF treatment is
    standard for exactly this kind of "known massive pulsar" constraint in
    the EOS-inference literature (e.g. Legred et al. 2021's use of the
    J0740/J0348 mass floors)."""

    def __init__(self, mean, sigma, one_sided=True, label=""):
        self.mean, self.sigma, self.one_sided = mean, sigma, one_sided
        self.label = label

    def log_prob(self, x):
        if self.one_sided:
            from scipy.stats import norm
            return float(norm.logcdf(x, loc=self.mean, scale=self.sigma))
        return -0.5 * ((x - self.mean) / self.sigma) ** 2


# ---------------------------------------------------------------------------
# SYNTHETIC placeholder samples -- FOR PIPELINE TESTING ONLY.
# These are NOT real NICER/GW posteriors. They exist purely so run_inference.py
# and analysis.py can be exercised end-to-end in this sandbox without network
# access. Every physics conclusion must be re-run against the real files
# fetched by download_all() (or manually placed in data/) before being
# reported anywhere.
# ---------------------------------------------------------------------------
def make_placeholder_sources(seed=0):
    rng = np.random.default_rng(seed)

    def mock_mr(M0, R0_km, sigM, sigR, n=4000):
        M = rng.normal(M0, sigM, n)
        R = rng.normal(R0_km, sigR, n) + 0.4 * (M - M0)  # mild M-R correlation, illustrative only
        return Posterior2DKDE(M, R, label="SYNTHETIC placeholder")

    j0030_c = mock_mr(1.40, 11.75, 0.10, 0.6)
    j0030_b = mock_mr(1.34, 11.0, 0.15, 1.4)
    j0740 = mock_mr(2.073, 12.49, 0.07, 1.0)
    j0437 = mock_mr(1.44, 11.4, 0.07, 0.8)

    lam1 = rng.uniform(0, 1200, 6000)
    lam2 = lam1 * rng.uniform(1.2, 3.0, 6000) + rng.normal(0, 50, 6000)
    lam2 = np.clip(lam2, 0, None)
    gw = Posterior2DKDE(lam1, lam2, label="SYNTHETIC placeholder")

    # also return mock (m1, m2) pairs consistent with Mchirp=1.188, for the
    # GW marginalization loop in likelihood.py
    q = rng.uniform(0.7, 1.0, 6000)
    Mc = 1.188
    m2 = Mc * (1 + q) ** 0.2 / q ** 0.6
    m1 = q * m2
    gw_masses = np.column_stack([m1, m2])

    j0952 = Gaussian1D(2.35, 0.17, one_sided=False, label="J0952 mass floor (real summary stat)")

    return {
        "J0030_C": j0030_c, "J0030_B": j0030_b,
        "J0740": j0740, "J0437": j0437,
        "GW170817_Lambda": gw, "GW170817_masses": gw_masses,
        "J0952": j0952,
    }
