"""
validate_solver.py -- Stage D. Before any hybrid EOS posterior is trusted,
prove the fast in-memory build_curve()/forward_map() interface agrees with
the ORIGINAL, already-validated (FSUGarnet Lambda_1.4=623 benchmark)
file-based EOStoObservables() on the same EOS instances.

This writes each in-memory hybrid table out to a temp file just so the
ORIGINAL solver (which only reads from disk) can run on it -- this file
round-trip is fine here because validation is a one-time, small-N check,
NOT the per-likelihood-call hot path (that's exactly the distinction that
motivated building solver_interface.py in the first place).
"""

import os
import tempfile
import numpy as np

import hybrid_eos
import solver_interface
from priors import make_prior_transform
from neutronstar_solver import EOStoObservables


def _write_temp_eos_file(P_arr, eps_arr):
    """numd (baryon number density) is not constructed for the hybrid
    branch (see hybrid_eos.py docstring, assumption #4) and is also
    confirmed UNUSED by EOStoObservables' actual physics -- filled with
    zeros here purely so the original file-reader's column layout is
    satisfied."""
    fd, path = tempfile.mkstemp(suffix=".dat")
    os.close(fd)
    order = np.argsort(P_arr)
    with open(path, "w") as f:
        f.write("# validation temp file\n# col0=eps col1=P col2=n(unused)\n#\n#\n")
        for p, e in zip(P_arr[order], eps_arr[order]):
            f.write(f"{e:.10e} {p:.10e} 0.0\n")
    return path


def run_validation(hadronic, n_draws=200, seed=0, r_cut_km=50.0):
    rng = np.random.default_rng(seed)
    prior_transform = make_prior_transform(hadronic)

    rows = []
    n_skipped = 0
    for i in range(n_draws):
        u = rng.uniform(size=6)
        theta = dict(zip(
            ["P_trans", "delta_p_frac", "d_eps_frac", "gamma2", "d_eps_core", "gamma3"],
            prior_transform(u)))
        table = hybrid_eos.build_hybrid_eos(hadronic, theta)
        if not table["valid"]:
            n_skipped += 1
            continue

        # "truth": original full-sweep, file-based solver
        path = _write_temp_eos_file(table["P"], table["eps"])
        try:
            truth = EOStoObservables(path, r_cut_km=r_cut_km)
        except Exception as e:
            n_skipped += 1
            os.remove(path)
            continue
        os.remove(path)

        # fast path
        curve = solver_interface.build_curve(table["P"], table["eps"], r_cut_km=r_cut_km)
        if not curve["ok"]:
            n_skipped += 1
            continue

        R_fast = float(curve["R_of_M"](1.4)) if curve["M_range"][0] <= 1.4 <= curve["M_range"][1] else np.nan
        L_fast = float(curve["Lambda_of_M"](1.4)) if curve["M_range"][0] <= 1.4 <= curve["M_range"][1] else np.nan

        rows.append({
            "R1.4_truth": truth["R1.4"], "R1.4_fast": R_fast,
            "Lambda1.4_truth": truth["Lambda1.4"], "Lambda1.4_fast": L_fast,
            "Mmax_truth": truth["Mmax"], "Mmax_fast": curve["Mmax"],
        })

    return rows, n_skipped


def summarize(rows):
    """MAE and max relative error across all three headline quantities
    pooled together, PLUS the mean relative error broken out per quantity
    (both requested explicitly)."""
    import pandas as pd
    df = pd.DataFrame(rows).dropna()

    quantities = ["R1.4", "Lambda1.4", "Mmax"]
    all_abs_err, all_rel_err = [], []
    per_quantity = {}

    for q in quantities:
        truth = df[f"{q}_truth"].values
        fast = df[f"{q}_fast"].values
        abs_err = np.abs(fast - truth)
        rel_err = abs_err / np.abs(truth)
        all_abs_err.append(abs_err)
        all_rel_err.append(rel_err)
        per_quantity[q] = {
            "mean_relative_error": float(np.mean(rel_err)),
            "max_relative_error": float(np.max(rel_err)),
            "mean_absolute_error": float(np.mean(abs_err)),
        }

    pooled_abs = np.concatenate(all_abs_err)
    pooled_rel = np.concatenate(all_rel_err)

    return {
        "n_valid": len(df),
        "pooled_MAE": float(np.mean(pooled_abs)),
        "pooled_max_relative_error": float(np.max(pooled_rel)),
        "per_quantity": per_quantity,
    }


if __name__ == "__main__":
    print("This module is meant to be driven from a script that has a real "
          "HadronicBaseline loaded (GM1/TM1/DD2/etc table). See run_inference.py "
          "for the intended usage pattern.")
