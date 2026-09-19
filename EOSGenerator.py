# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %%
"""
Generates neutron-star EOS tables: a tabulated crust stitched to a
3-segment polytrope built directly in cs^2(n) space, with the two
internal segment boundaries smoothly blended (no derivative kink).

Output format matches what EOStoObservables[_parallel] expects:
    4 header lines, then columns
    energy_density[MeV/fm^3]   pressure[MeV/fm^3]   number_density[fm^-3]
strictly increasing in both energy density and pressure.

Crust: supply your own tabulated (n, eps, P) file -- e.g. pulled from
CompOSE (compose.obspm.fr) or the Douchin & Haensel (2001) SLy4 table.
Don't derive it from scratch; crust choice barely affects M-R/Lambda.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp

N_SAT = 0.16  # fm^-3, nuclear saturation density -- reference scale for priors


def load_crust(path):
    """path: whitespace table, columns n[fm^-3] eps[MeV/fm^3] P[MeV/fm^3]."""
    n, eps, P = np.loadtxt(path, unpack=True)
    order = np.argsort(n)
    n, eps, P = n[order], eps[order], P[order]
    logn = np.log(n)
    eps_of_logn = PchipInterpolator(logn, np.log(eps))
    P_of_logn = PchipInterpolator(logn, np.log(P))
    eps_fn = lambda nn: np.exp(eps_of_logn(np.log(nn)))
    P_fn = lambda nn: np.exp(P_of_logn(np.log(nn)))
    return n, eps, P, eps_fn, P_fn


def segment_state(n, n0, eps0, P0, Gamma):
    """Exact closed form for a single polytrope P = K n^Gamma anchored at
    (n0, eps0, P0) -- Read et al. (2009) construction. Returns eps, P, cs2."""
    P = P0 * (n / n0) ** Gamma
    a = eps0 / n0 - P0 / (n0 * (Gamma - 1))
    eps = a * n + P / (Gamma - 1)
    cs2 = Gamma * P / (eps + P)
    return eps, P, cs2


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def sample_params(rng, n_cc=0.5 * N_SAT):
    """Physically-motivated (uniform) priors. Widen G_LO toward 0 if you
    want to expose the network to phase-transition-like softening; keep
    it away from Gamma=1, where the closed form is singular."""
    n1 = rng.uniform(1.0, 3.0) * N_SAT
    n2 = rng.uniform(3.5, 7.0) * N_SAT
    f1, f2 = rng.uniform(0.05, 0.3, size=2)
    G0, G1, G2 = rng.uniform(1.4, 5.0, size=3)
    return dict(n_cc=n_cc, n1=n1, n2=n2, w1=f1 * n1, w2=f2 * n2,
                G0=G0, G1=G1, G2=G2)


def build_cs2_fn(params, crust_eps_fn, crust_P_fn):
    """Returns cs2(n) for the whole core: three segments in cs^2(n),
    smoothstep-blended across [n_i - w_i/2, n_i + w_i/2]."""
    n_cc, n1, n2 = params["n_cc"], params["n1"], params["n2"]
    w1, w2 = params["w1"], params["w2"]
    G0, G1, G2 = params["G0"], params["G1"], params["G2"]

    anchor0 = (n_cc, float(crust_eps_fn(n_cc)), float(crust_P_fn(n_cc)))
    e1, P1, _ = segment_state(n1, *anchor0, G0)
    anchor1 = (n1, e1, P1)
    e2, P2, _ = segment_state(n2, *anchor1, G1)
    anchor2 = (n2, e2, P2)

    def cs2(n):
        if n < n1 - w1 / 2:
            _, _, c = segment_state(n, *anchor0, G0)
        elif n < n1 + w1 / 2:
            _, _, ca = segment_state(n, *anchor0, G0)
            _, _, cb = segment_state(n, *anchor1, G1)
            s = smoothstep((n - (n1 - w1 / 2)) / w1)
            c = (1 - s) * ca + s * cb
        elif n < n2 - w2 / 2:
            _, _, c = segment_state(n, *anchor1, G1)
        elif n < n2 + w2 / 2:
            _, _, ca = segment_state(n, *anchor1, G1)
            _, _, cb = segment_state(n, *anchor2, G2)
            s = smoothstep((n - (n2 - w2 / 2)) / w2)
            c = (1 - s) * ca + s * cb
        else:
            _, _, c = segment_state(n, *anchor2, G2)
        return min(max(c, 1e-6), 0.999999)  # causality/stability clip

    return cs2


def generate_eos(rng, crust_path, n_max=10 * N_SAT, n_points=2000, out_path=None):
    n_crust, eps_crust, P_crust, crust_eps_fn, crust_P_fn = load_crust(crust_path)
    params = sample_params(rng)
    n_cc = params["n_cc"]
    assert n_cc <= n_crust.max(), "n_cc beyond crust table range -- extend the crust file"
    cs2 = build_cs2_fn(params, crust_eps_fn, crust_P_fn)

    # Integrate the first law directly: deps/dn = (eps+P)/n, dP/dn = cs2*deps/dn.
    # This is what actually enforces global thermodynamic consistency --
    # the segment closed forms above only define cs2(n), not the final eps,P.
    def rhs(n, y):
        deps_dn = (y[0] + y[1]) / n
        return [deps_dn, cs2(n) * deps_dn]

    y0 = [float(crust_eps_fn(n_cc)), float(crust_P_fn(n_cc))]
    n_core = np.logspace(np.log10(n_cc), np.log10(n_max), n_points)
    n_core[0] = n_cc  # avoid a log10/10** round-trip mismatch with t_span
    sol = solve_ivp(rhs, (n_core[0], n_core[-1]), y0, t_eval=n_core, rtol=1e-9, atol=1e-12)
    if not sol.success:
        return None, params

    n_lo = n_crust[n_crust < n_cc]
    n_all = np.concatenate([n_lo, n_core])
    eps_all = np.concatenate([eps_crust[n_crust < n_cc], sol.y[0]])
    P_all = np.concatenate([P_crust[n_crust < n_cc], sol.y[1]])

    if not (np.all(np.diff(eps_all) > 0) and np.all(np.diff(P_all) > 0)):
        return None, params  # reject -- numerically pathological draw

    if out_path is not None:
        header = (
            f"n1={params['n1']:.5f} n2={params['n2']:.5f} w1={params['w1']:.5f} "
            f"w2={params['w2']:.5f}\nG0={params['G0']:.4f} G1={params['G1']:.4f} "
            f"G2={params['G2']:.4f} n_cc={n_cc:.5f}\n"
            "columns: energy_density[MeV/fm^3]  pressure[MeV/fm^3]  number_density[fm^-3]\n"
            "generated by generate_eos()"
        )
        np.savetxt(out_path, np.column_stack([eps_all, P_all, n_all]), header=header)

    return (eps_all, P_all, n_all), params


def cs2_on_sensor_grid(params, crust_eps_fn, crust_P_fn, sensor_grid):
    """Evaluate the same cs2(n) on a FIXED grid shared across every
    generated EOS -- this, not the integration grid above, is the
    DeepONet branch-net input."""
    cs2 = build_cs2_fn(params, crust_eps_fn, crust_P_fn)
    return np.array([cs2(n) for n in sensor_grid])


if __name__ == "__main__":
    import csv

    rng = np.random.default_rng(0)
    crust_path = "sly4_crust_eos"  # replace with your CompOSE / DH2001 table
    n_ok = 0
    with open("params.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "n1", "n2", "w1", "w2", "G0", "G1", "G2", "n_cc"])
        while n_ok < 20:
            out_path = f"eos_{n_ok:05d}.dat"
            result, params = generate_eos(rng, crust_path, out_path=out_path)
            if result is None:
                continue
            writer.writerow([out_path] + [params[k] for k in
                             ["n1", "n2", "w1", "w2", "G0", "G1", "G2", "n_cc"]])
            n_ok += 1
    print(f"Wrote {n_ok} EOS files + params.csv")

# %%
