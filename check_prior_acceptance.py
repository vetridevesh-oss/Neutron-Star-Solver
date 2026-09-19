"""
check_prior_acceptance.py -- run this FIRST, before any real inference,
against your actual hadronic table (GM1/TM1/DD2/NL3wr4). Empirically
measures what fraction of naive uniform prior draws survive the
causality/stability veto in hybrid_eos.py.

WHY THIS MATTERS (found empirically while building this pipeline, not
guessed): with the K2/K3 = P_anchor/eps_ns^gamma normalization used in
hybrid_eos.py, random draws from the given 6-parameter uniform priors were
acausal roughly 90-95% of the time on a synthetic placeholder table (max
cs^2 sometimes in the thousands). This MAY be expected/fine -- broad
uniform priors on crossover stiffness are known in the field to put most of
their prior mass in unphysical regions, which is exactly why the causality
veto exists, and dynesty's nested sampling (not naive rejection sampling)
is specifically built to find the valid sub-manifold efficiently. But it
could also mean the K-normalization convention (flagged as a genuine
assumption in hybrid_eos.py's docstring, since the spec doesn't pin it
down) needs revisiting for YOUR real hadronic table. Run this and look at
the acceptance fraction before trusting a full run:
  - a few percent: normal, dynesty should handle it, just budget more
    wall-clock time for the live-point initialization phase
  - <<1%: likely a real problem -- revisit the K2/K3 convention (see the
    "Core segment transition energy density" section of hybrid_eos.py) or
    narrow the gamma2/gamma3 prior ranges
"""
import numpy as np
from hybrid_eos import HadronicBaseline, build_hybrid_eos
from priors import make_prior_transform, theta_array_to_dict


def check_acceptance(hadronic_path, n_draws=5000, seed=0):
    hadronic = HadronicBaseline(hadronic_path)
    pt = make_prior_transform(hadronic)
    rng = np.random.default_rng(seed)

    n_valid = 0
    reject_reasons = {}
    for _ in range(n_draws):
        u = rng.uniform(size=6)
        theta = theta_array_to_dict(pt(u))
        table = build_hybrid_eos(hadronic, theta)
        if table["valid"]:
            n_valid += 1
        else:
            key = table["reject_reason"].split(":")[0]
            reject_reasons[key] = reject_reasons.get(key, 0) + 1

    print(f"Hadronic table: {hadronic_path}")
    print(f"Acceptance fraction: {n_valid}/{n_draws} = {100*n_valid/n_draws:.2f}%")
    print("Rejection breakdown:")
    for k, v in sorted(reject_reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {k}: {v} ({100*v/n_draws:.1f}%)")
    return n_valid / n_draws


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "EOSBetaNL3.dat"
    check_acceptance(path)
