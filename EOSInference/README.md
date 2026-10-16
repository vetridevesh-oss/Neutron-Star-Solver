# Two-transition hybrid-core Bayesian inference

```
hadronic table  ->  1st-order jump  ->  polytrope 1  ->  1st-order jump  ->  polytrope 2
                                        P = K eps^gamma (energy-density polytropes)
```

Sampled parameters: `eps_trans`, `delta_eps_trans`, `eps_1` (top of polytrope 1,
pre-jump side), `delta_eps_1`, `gamma1`, `gamma2`. Everything you may want to
change is in `config.py`.

**Read `AUDIT.md` first.** It records what was wrong, what is now verified
against published numbers, and the physics decisions that are still open — two
of which change every Bayes factor.

| file | role |
|---|---|
| `config.py` | prior ranges, sweep tolerances, data manifest (path + md5 + DOI + columns), branch rule, sampler settings |
| `hybrid_eos.py` | hadronic table (log-log Pchip), the two polytropes and jumps, the c_s² envelope, and the Maxwell `n(eps)` / `mu(P)` reconstruction |
| `solver.py` | TOV + y solver, numba-compiled; Takátsy & Kovács (2020) Eq. 11 jump; noise-aware stable-branch decomposition |
| `datasets.py` | provenance-checked loaders, weighted FFT KDE, headline reproduction tests, sampling-prior forms |
| `priors.py` | unit cube -> the six parameters |
| `likelihood.py` | NICER (M,R) terms, GW170817 in (q, Λ1, Λ2) at fixed chirp mass, J0952 mass term, branch rules |
| `model.py` | prior + EOS + solver + likelihood -> callables for dynesty |
| `run_inference.py` | dynesty driver, one run per hadronic EOS, ln B against the pure hadronic EOS |
| `analysis.py` | posterior summary, corner plot, M-R band, branch and transition-reachability diagnostics |
| `validate.py` | 9 groups of numerical checks; exits non-zero if any fail |
| `mockdata.py` | synthetic data built as real likelihood *terms*, so mock runs use the production code path |

## Run

```bash
pip install numpy scipy numba dynesty corner matplotlib mpmath h5py
python validate.py                                            # ~6 min, --quick for ~2
python run_inference.py --mock --nlive 80 --maxcall 4000       # pipeline test, no real data
python run_inference.py --nlive 500 --nprocs 8                 # real data
python run_inference.py --rule max                             # branch rule: mixture | max | sum
python analysis.py results/EOSBetaFSU2L80
```

## Data

`config.DATA` carries one entry per source with its **own column map**, md5 and
DOI. The column order is not the same across releases — J0740 and J0030 are
(weight, M, R) while the Miller J0437 file is (R, M, weight) — so each source
also declares its published headline numbers, and `datasets.check_headline`
re-derives them from the loaded arrays when the likelihood is built. A
transposition or a wrong weight column fails loudly instead of producing a
plausible wrong answer.

GW170817 is read from the GWTC-1 HDF5 directly; the detector-frame masses are
converted with a configurable redshift.

## Things to know

* **The likelihood is an integral against a population prior**, not a density
  evaluated at a point. Each pulsar term is
  `L = (1/|U|) ∫ dm K(m, R(m;θ))` over the EOS's own stable mass support `U`,
  so 1/|U| is an Occam penalty on large M_TOV. That choice matters — see
  `AUDIT.md` §4.1.
* **GW170817** uses the released posterior's sampling prior explicitly: Λ1 and
  Λ2 are flat on [0, 5000] (verified from the release's prior samples) but q is
  *not*. With the EOS-side population prior also uniform in component masses the
  q-shapes cancel inside the integral and survive in the normaliser. Because the
  flat-Λ1Λ2 prior is kept rather than reweighted, Λ̃ from these samples
  (median 406, 90% upper 793) does not match the familiar "Λ̃ ≤ 720".
* **Branch rules are different priors over (mass, branch)**, not different
  conventions. `mixture` and `sum` are both proper marginals with their own
  normalisers; `max` is a profile and is not a likelihood.
* **Hard causality** is enforced on a sub-sampled c_s² envelope, not at table
  knots (which under-reported by up to 14×), and only over pressures occupied by
  stars on *stable* branches.
* **Prior reparameterisations change the evidence.** Fractional vs absolute
  `delta_eps`, the conditional `eps_1`, and `eps1_cap` all move ln B. Record
  which you used; `summary.json` stores the whole prior and likelihood config.
* **The pure-hadronic baseline has no free parameters**, so Z_had = L_had and
  ln B = ln Z_hybrid − ln L_had.
