"""
compare.py -- the cross-EOS figure: ln B for every hadronic table in one view.

    python compare.py results/*            # or: python compare.py results

This is the result of the study, so it is deliberately a separate script from
analysis.py: it reads the finished summary.json of each run and adds nothing to
them.  For every table it also solves the PURE HADRONIC sequence once, which is
cheap and gives the physical x-axis (R_1.4 and M_TOV of the baseline) that a
reader actually wants -- ln B against a model label means nothing.

The FSU2L family is a sweep in the slope of the symmetry energy L, which the
file name carries (EOSBetaFSU2L80 -> L = 80 MeV), so that family gets a
continuous axis; every other table is drawn as a labelled point.

Writes lnB_summary.png and lnB_table.csv next to the result directories.
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

import config
import solver
from hybrid_eos import HadronicTable, HybridEOS

M_REF = 1.4


def baseline(eos_file):
    """M_TOV and R_1.4 of the pure hadronic table."""
    tab = HadronicTable(eos_file, config.EOS_SKIPROWS, config.EOS_COLS)
    seq = solver.compute_sequence(HybridEOS.pure_hadronic(tab))
    R = np.nan
    for b in seq.branches:
        if b["M"][0] <= M_REF <= b["M"][-1]:
            R = float(b["interp"]["R"](M_REF))
    return float(seq.M_tov), R


def collect(dirs):
    rows = []
    for d in dirs:
        f = os.path.join(d, "summary.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f))
        name = os.path.basename(os.path.normpath(d))
        m = re.search(r"FSU2L(\d+)", name)
        Mtov, R14 = baseline(s["eos"])
        rows.append(dict(name=name, eos=s["eos"],
                         L=float(m.group(1)) if m else np.nan,
                         lnB=float(s["lnB_hybrid_vs_hadronic"]),
                         err=float(s.get("lnB_err", s["logZ_err"])),
                         lnZ=float(s["logZ"]), lnL_had=float(s["lnL_hadronic"]),
                         f_acc=float((s.get("prior_acceptance") or {}).get("f_acc", np.nan)),
                         M_tov_had=Mtov, R14_had=R14))
    return sorted(rows, key=lambda r: (np.isnan(r["L"]), r["L"], r["name"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="result directories, or one parent directory")
    ap.add_argument("--outdir", default=None, help="where to write (default: the parent)")
    args = ap.parse_args()

    dirs = args.dirs
    if len(dirs) == 1 and not os.path.exists(os.path.join(dirs[0], "summary.json")):
        dirs = sorted(glob.glob(os.path.join(dirs[0], "*")))
    rows = collect(dirs)
    if not rows:
        raise SystemExit("no summary.json found in: %s" % ", ".join(args.dirs))
    out = args.outdir or os.path.dirname(os.path.normpath(dirs[0])) or "."

    print(f"{'table':>22} {'L':>5} {'lnB':>8} {'+/-':>5} {'M_TOV_had':>10} {'R1.4_had':>9} {'f_acc':>6}")
    for r in rows:
        print(f"{r['name']:>22} {r['L']:5.0f} {r['lnB']:+8.2f} {r['err']:5.2f} "
              f"{r['M_tov_had']:10.3f} {r['R14_had']:9.2f} {r['f_acc']:6.3f}")

    with open(os.path.join(out, "lnB_table.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("figure skipped (matplotlib unavailable):", e)
        return

    fam = [r for r in rows if np.isfinite(r["L"])]
    oth = [r for r in rows if not np.isfinite(r["L"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))

    if fam:
        a1.errorbar([r["L"] for r in fam], [r["lnB"] for r in fam],
                    yerr=[r["err"] for r in fam], marker="o", ms=4, lw=1.2,
                    color="C0", capsize=2, label="FSU2L family")
    a1.axhline(0, color="k", lw=1)
    a1.axhspan(-1, 1, color="0.85", zorder=0, label="|ln B| < 1 (inconclusive)")
    a1.set_xlabel("L [MeV]  (slope of the symmetry energy)")
    a1.set_ylabel(r"$\ln B$(hybrid / hadronic)")
    a1.legend(fontsize=9)

    for r in rows:
        a2.errorbar(r["R14_had"], r["lnB"], yerr=r["err"], marker="o", ms=5,
                    color="C0" if np.isfinite(r["L"]) else "C3", capsize=2)
    for r in oth:
        a2.annotate(r["name"].replace("EOSBeta", "").replace("EOS", ""),
                    (r["R14_had"], r["lnB"]), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    a2.axhline(0, color="k", lw=1)
    a2.axhspan(-1, 1, color="0.85", zorder=0)
    a2.set_xlabel(r"$R_{1.4}$ of the hadronic baseline [km]")
    a2.set_ylabel(r"$\ln B$(hybrid / hadronic)")

    plt.tight_layout()
    plt.savefig(os.path.join(out, "lnB_summary.png"), dpi=120, bbox_inches="tight")
    print("wrote lnB_summary.png, lnB_table.csv in", out)


if __name__ == "__main__":
    main()
