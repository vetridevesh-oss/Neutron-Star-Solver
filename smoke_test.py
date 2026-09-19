"""
smoke_test.py -- proves the pipeline runs end-to-end. Uses:
  (a) a SYNTHETIC hadronic table (a simple stiff polytrope, NOT GM1/TM1/DD2)
  (b) SYNTHETIC placeholder observational sources (fetch_observational_data.make_placeholder_sources)
This is NOT a physics result. It exists to catch wiring/shape/unit bugs
before your mentee plugs in the real hadronic table + real fetched data.
"""
import os
import numpy as np

os.chdir(os.path.dirname(__file__))

from neutronstar_solver import P0

# --- (a) build a throwaway "hadronic-like" table just to exercise the code ---
n = np.logspace(-3, 0.3, 400)          # fm^-3, crust to a few n0
P = 20.0 * n ** 2.6                     # MeV/fm^3, a simple stiff-ish polytrope
eps = 939.0 * n + P / 1.6               # MeV/fm^3, roughly rest mass + internal energy
tab_path = "data/_placeholder_hadronic.dat"
os.makedirs("data", exist_ok=True)
order = np.argsort(P)
with open(tab_path, "w") as f:
    f.write("# PLACEHOLDER hadronic table -- NOT GM1/TM1/DD2/NL3wr4\n#\n#\n#\n")
    for p, e, nn in zip(P[order], eps[order], n[order]):
        f.write(f"{e:.8e} {p:.8e} {nn:.8e}\n")

from hybrid_eos import HadronicBaseline, build_hybrid_eos
hadronic = HadronicBaseline(tab_path)
print(f"[ok] loaded placeholder hadronic table: P in [{hadronic.P_min:.2f}, {hadronic.P_max:.2f}] MeV/fm^3")

from priors import make_prior_transform, theta_array_to_dict
pt = make_prior_transform(hadronic)
rng = np.random.default_rng(1)
u = rng.uniform(size=6)
theta = theta_array_to_dict(pt(u))
print(f"[ok] sampled theta: { {k: round(v,3) for k,v in theta.items()} }")

table = build_hybrid_eos(hadronic, theta)
print(f"[ok] built hybrid EOS table, valid={table['valid']} reason='{table['reject_reason']}'")

import solver_interface
if table["valid"]:
    curve = solver_interface.build_curve(table["P"], table["eps"])
    print(f"[ok] build_curve: ok={curve['ok']}", curve.get("reason", ""))
    if curve["ok"]:
        print(f"     Mmax={curve['Mmax']:.3f} Msun, stable M range={curve['M_range']}")
        if curve["M_range"][0] <= 1.4 <= curve["M_range"][1]:
            print(f"     R(1.4)={float(curve['R_of_M'](1.4)):.3f} km  "
                  f"Lambda(1.4)={float(curve['Lambda_of_M'](1.4)):.1f}")

# --- (b) placeholder observational sources + one likelihood evaluation ---
from fetch_observational_data import make_placeholder_sources
sources = make_placeholder_sources()
print("[ok] built placeholder observational sources:", list(sources.keys()))

from likelihood import make_log_likelihood, hadronic_null_loglike
loglike = make_log_likelihood(hadronic, sources, nicer_scenario="C", n_scan=60)
theta_arr = pt(u)
ll = loglike(theta_arr)
print(f"[ok] log_likelihood(theta) = {ll}")

ll_had = hadronic_null_loglike(hadronic, sources, nicer_scenario="C", n_scan=60)
print(f"[ok] hadronic-null log-likelihood = {ll_had}")

# --- tiny end-to-end dynesty run (few live points, just to prove wiring) ---
import dynesty
sampler = dynesty.NestedSampler(loglike, pt, ndim=6, nlive=25)
sampler.run_nested(dlogz=2.0, print_progress=False)
res = sampler.results
print(f"[ok] tiny dynesty run completed: logZ = {res.logz[-1]:.2f} +/- {res.logzerr[-1]:.2f}, "
      f"{len(res.samples)} samples")

import pickle
os.makedirs("results", exist_ok=True)
out = {"label": "PLACEHOLDER", "nicer_scenario": "C", "logZ_had": ll_had,
       "logZ_hybrid": float(res.logz[-1]), "logZ_hybrid_err": float(res.logzerr[-1]),
       "log_bayes_factor": float(res.logz[-1]) - ll_had,
       "samples": res.samples, "logwt": res.logwt, "logz": res.logz,
       "param_names": ["P_trans", "delta_p_frac", "d_eps_frac", "gamma2", "d_eps_core", "gamma3"]}
with open("results/PLACEHOLDER_C.pkl", "wb") as f:
    pickle.dump(out, f)

from analysis import credible_intervals, correlation_matrix, make_corner_plot, bayes_factor_table, stress_test_reach
ci_df, samples_eq = credible_intervals(out)
print("[ok] credible intervals:\n", ci_df.to_string(index=False))

corr = correlation_matrix(samples_eq)
print("[ok] correlation matrix:\n", corr.round(2).to_string())

make_corner_plot(out, "results/PLACEHOLDER_corner.png")
print("[ok] corner plot saved")

bf_table = bayes_factor_table({"PLACEHOLDER_C": out})
print("[ok] Bayes factor table:\n", bf_table.to_string(index=False))

stress_df = stress_test_reach(hadronic, samples_eq, n_check=25)
print("[ok] HESS/XTE stress-test diagnostic:\n", stress_df.to_string(index=False))

print("\nSMOKE TEST COMPLETE -- all stages ran without error on placeholder data.")
