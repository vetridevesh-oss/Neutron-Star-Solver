# Audit of the two-transition hybrid-core inference package

Date: 2026-10-15. Baseline EOS for all numbers below: `EOSBetaFSU2L80.dat`.
Everything quoted here was measured by running the code, not asserted.

Reproduce with `python validate.py` (about 6 min; `--quick` for 2).

---

## 1. Verdict

The TOV + tidal solver is correct and is now validated against an external
number, not only against itself: **FSUGarnet gives Λ₁.₄ = 622.9 against the
published 623, and M_TOV = 2.066 against 2.07 ± 0.02** (Piekarewicz & Fattoyev
2019; Chen & Piekarewicz 2015). NL3 (2.774, R₁.₄ = 14.81) and IUFSU (1.939) also
land where the literature puts them. The Takátsy–Kovács jump condition is right
and the derivation checks out.

Four things were wrong, two of them seriously:

| | Severity | Status |
|---|---|---|
| Stable-branch detection was grid-dependent | **high** — M_TOV moved by 3× on one EOS | fixed |
| Causality screen under-reported c_s² by up to **14×** | **high** — acausal EOSs passed | fixed |
| GW170817 term was not a likelihood | **high** — known gap | rewritten |
| ε₁ prior did not match the approved spec | medium — changes every Bayes factor | fixed, flagged |

Two physics issues are **decisions you have to make**, not bugs; they are in §4.
Nothing in the package is ready to produce a publishable Bayes factor until
those are settled, because each one moves ln B.

---

## 2. Bugs found and fixed

### 2.1 Branch detection was not grid-robust (`solver.py`)

`_extrema_insert` refined around every extremum of M(P_c) on every round, with
no stopping rule. After a few rounds it had bracketed the extrema so tightly
that consecutive central pressures differed by Δln P_c ≈ 8×10⁻⁵, where the mass
difference is **ΔM ~ 10⁻⁹ M_⊙ — below the integrator's own error**. The sign of
dM then flips at random. The turning-point counter incremented on each flip, and
`cur = max(cur, 0)` silently absorbed the resulting bookkeeping error instead of
failing, so the count never returned to zero and every subsequent branch was
labelled unstable.

Measured on one twin EOS from the prior:

```
default grid : 2 branches, M_TOV = 1.281
dense grid   : 1 branch,   M_TOV = 0.410      <- same EOS, different grid
```

Seven spurious extrema in a row appeared at P_c ≈ 4.52×10⁻³ with |ΔM| between
2×10⁻⁹ and 1.3×10⁻⁸.

**Fix.** Three changes:

1. The turning-point counter is gone. Along a sequence ordered by central
   pressure, extrema of M alternate max/min and the sequence starts stable, so
   the Bardeen–Thorne–Meltzer mode count is 0 on every rising segment and 1 on
   every falling one — the counting *is* `dM/dP_c > 0`. Same criterion, no state
   to corrupt.
2. A dead-band: segments whose |ΔM| is at the ODE noise floor
   (`noise_factor × rtol × M_max`, default 50 × 10⁻⁶ × M) carry no information
   and inherit the sign of the nearest resolved segment to their left.
3. A resolution floor `min_dlnPc = 10⁻³`: refinement stops there, the transition
   clustering no longer goes below it (the 3×10⁻⁴ entry was dropped), and the
   grid is thinned to it before the branch analysis.

**Verification.** All four twin EOSs now agree on branch count and M_TOV to
≤ 10⁻⁴ across three grids spanning n_base 70–600 and rtol 10⁻⁶–10⁻⁸. Note this
was never visible in the single-branch, high-mass region that dominates the
posterior — there the solver already agreed to ΔM_TOV ~ 10⁻⁷, ΔR₁.₄ ~ 4×10⁻³ km,
ΔΛ/Λ ~ 3×10⁻⁴ — but twin and third-family branches are the point of the study.

### 2.2 The causality screen under-reported c_s² (`hybrid_eos.py`)

`cs2_max_below` took the running maximum of c_s² over the **table knots only**.
A cubic Hermite segment can have a smaller d(ln ε)/d(ln P), and hence a larger
c_s², strictly inside an interval than at either end. Brute force at 256
sub-samples per interval against the old knots-only screen:

```
max(brute force / knots only) = 14.33
```

So an EOS that is acausal between knots below ε_trans passed the screen. On
FSU2L80 the true maximum c_s² in the table is 0.99998 — the table runs right up
against the causal limit at its top end, which is exactly where ε_trans is
allowed to sit (the prior reaches 6 n₀). The screen's error was therefore in the
region that matters.

**Fix.** A c_s² envelope is built once per table on 64 sub-samples per interval,
and the lookup rounds the index **up** so the sub-interval containing P is always
included — a safety screen must never under-report. Residual: the envelope is now
within 4×10⁻⁵ of brute force.

Side effect worth knowing: the maximum c_s² reached inside the stable hadronic
FSU2L80 stars is 0.462, not the 0.415 the old screen reported.

### 2.3 GW170817 term (`likelihood.py`)

The old term evaluated a 2-D (Λ1, Λ2) KDE — already marginalised over mass — at
Λ values computed from posterior (m1, m2) pairs, and averaged. That is not an
integral of anything: it reuses the mass information twice, divides out no
sampling prior, uses detector-frame masses as if they were source-frame, and
never imposes m1 ≤ M_max(θ).

**What it is now.** The detector-frame chirp mass is measured to 5.8×10⁻⁵
relative (measured from the samples), so the binary is parameterised by
(q, Λ1, Λ2) at fixed M_c. Writing the EOS-side population prior as uniform in
component masses — the same form as the sampling prior — the q-shapes cancel
inside the integral and survive only in the normaliser:

```
L_GW(θ) = (1/Z(θ)) ∫_{A(θ)} dq  p_post(q, Λ1(q;θ), Λ2(q;θ))
Z(θ)    = ∫_{A(θ)} dq  π_s(q),      π_s(q) ∝ q^(−6/5) (1+q)^(2/5)
```

with A(θ) the mass ratios for which both components sit on a stable branch. The
q-prior division is therefore **not** a no-op: it sets the Occam factor.

Three claims this rests on, all checked against the release's own prior samples
(`validate.py` test 8):

* Λ1 and Λ2 priors flat on [0, 5000]: flat to 4.2% and 6.5%, against a Poisson
  floor of 5.3%. Confirmed.
* π(q) ∝ q^(−6/5)(1+q)^(2/5) **at fixed M_c**: χ²/dof = 0.99 on 1050 prior
  samples in a ±10% M_c window. Confirmed. (Marginalised over the prior's full
  M_c range it deviates by 28%, which is why the window matters.)
* Masses are detector-frame. Converted with the host-galaxy redshift
  (z = 0.0099, configurable), which keeps M_c,src as sharp as M_c,det
  (5.8×10⁻⁵); the per-sample alternative inherits the distance uncertainty and
  widens it to 1.6×10⁻³. Sensitivity of ln L_GW: z = 0 → −16.04, z = 0.0099 →
  −16.20, z = 0.012 → −16.22, per-sample → −16.18. Sub-0.2 nat; not a worry.

**Λ̃ cross-check.** These samples give Λ̃ = 406 (median), 793 (90% upper). The
familiar "Λ̃ ≤ 720" is quoted after reweighting to a flat Λ̃ prior. This pipeline
deliberately uses the raw posterior with its flat Λ1, Λ2 prior divided out, so
the numbers should not match and the difference is the prior. Say this in the
paper before a referee says it.

### 2.4 Prior on ε₁ did not match the approved specification

The spec puts ε₁ "up to about 1800 MeV/fm³". The `"width"` mode implemented
ε₁ − start₁ ~ U[10, 1500] with no ceiling, which sent **54% of prior draws above
1800** (max 4361) and put the bottom of polytrope 2 beyond any central density a
stable star reaches. Over half the prior volume was therefore a one-transition
model wearing two-transition clothes, diluting Z_hybrid and inflating the Occam
penalty on the thing you are trying to measure.

**Fix.** `PRIOR["eps1_cap"] = 1800.0`, applied in both modes. Draws where
ε_trans + Δε_trans leaves no room below the cap now fall outside the support
(17.8% of the unit cube) and contribute zero, which is what a truncated prior
should do. Set it to `None` to recover the old behaviour — that changes every
Bayes factor, so record which you used.

### 2.5 The mass quadrature was not converged

`R(m)` and `Λ(m)` behave like √(m_max − m) at the end of a stable branch, because
dM/dP_c vanishes there. Their derivative is infinite at the endpoint, so a
uniform trapezoid rule converges only as n^(−3/2). At the configured
`n_mass_grid = 150` the J0740 term was off by **0.077 nat** and still drifting at
n = 2400:

```
n_mass_grid     150      300      600     1200     2400
J0740 ln L   -2.5665  -2.5260  -2.5039  -2.4941  -2.4892
```

**Fix.** The quadrature abscissae are now clustered quadratically at the ends of
the integration range, at each branch's own endpoints, and over the interval
where the data have support. With the same 150 points:

```
n_mass_grid      75      150      300      600     1200
J0740 ln L  -2.48504 -2.48451 -2.48463 -2.48464 -2.48469
J0030 ln L  -1.49477 -1.48958 -1.48898 -1.48891 -1.48892
J0437 ln L  -1.73228 -1.72629 -1.72622 -1.72599 -1.72602
```

Converged to ≤ 3×10⁻⁴ nat. The same clustering is applied to the GW mass-ratio
integral, where the m1 = M_TOV endpoint has the same behaviour whenever M_TOV
rather than m_min sets the lower limit on q. The whole likelihood now costs
0.45 ms per call against roughly 40 ms for the TOV sweep.

### 2.6 Smaller things

* **Column orders differ between your three NICER files.** J0740 and J0030 are
  (weight, M, R); the Miller J0437 file is (R, M, weight). Each source now
  declares its own map plus md5 and DOI, and the loader re-derives the published
  median and 68% interval at construction. All six checks pass. (The J0740
  values 2.073 ± 0.069 and 12.49 +1.28/−0.88 are genuine published headlines;
  the J0030 and J0437 reference values in `config.DATA` are derived from the
  files themselves, so they guard against a future transposition but **not**
  against the file being the wrong release — verify them against the papers.)
* **The validation EOS in tests 3–5 was acausal** (c_s² = 4.02 at the top of
  polytrope 1, 3.42 inside the star). It could never appear in a posterior, so
  the convergence and jump tests were measuring a region the sampler never
  visits. Replaced with a causal case that has a genuine stable hybrid core.
* **The jump test compared the wrong EOS.** Its smoothed reference blends only
  the *first* transition, but the test star had P_c = 3 P_t > P_1, so it also
  contained polytrope-2 matter the reference did not model. With P_c between P_t
  and P_1 the smoothed result now converges to the corrected TK form as it
  should (Λ = 225.91 smoothed vs 225.83 TK vs 226.02 PPL).
* **`"sum"` branch rule was normalised wrongly** — it divided by (M_TOV − m_min)
  like the mixture rule. It now divides by S = Σ_b (branch mass span), which is
  what makes it a probability statement rather than double counting.
* Weighted samples were being resampled down to 5000 with replacement and the
  weights thrown away. With ESS of 1.5×10⁵–10⁶ in your files that discarded
  almost everything. The KDE now keeps all samples and all weights, via FFT on a
  whitened grid (0.2 s to build; 1 ms per likelihood call for all four terms).
* Hard edges of the sampling priors (R ≤ 16 km for J0740 and J0030, M ≥ 1 for
  J0030, Λ ≥ 0, q ≤ 1) are handled by mirroring samples through them, rather
  than letting the kernel leak probability out of the support.

---

## 3. What is now verified

| Check | Result |
|---|---|
| Λ₁.₄, M_TOV vs published (FSUGarnet) | 622.9 vs 623; 2.066 vs 2.07 ± 0.02 |
| TOV + y vs independent scipy integration | ΔM/M ≤ 3×10⁻⁷, ΔΛ/Λ ≤ 1.7×10⁻⁴ |
| k₂ double precision vs mpmath, β ≥ 0.005 | ≤ 1.6×10⁻⁷ |
| Sharp TK jump vs thin smoothed transition | agree to 4×10⁻⁴ in Λ; PPL form off by 0.08% here |
| M_TOV convergence at production settings | 2.7×10⁻⁶ |
| Branch decomposition across three grids | identical |
| Causality envelope vs brute force | within 4×10⁻⁵, conservative |
| μ continuity across both jumps | 10⁻¹² relative |
| Δn ≥ 0 across both jumps | holds by construction |
| First law d(ε/n)/d(1/n) = −P in the polytropes | residual 7×10⁻⁴ (finite-difference limited) |
| NICER headline reproduction (6 numbers) | all pass |
| GW sampling prior: flat Λ, π(q) form | both confirmed |
| ln L stability, 7× bandwidth range (NICER) | < 0.06 nat |
| ln L stability, split-half of samples | < 0.03 nat (NICER), 0.08 (GW) |
| M_TOV never truncated by the sweep ceiling | 0 of 40 draws |
| Mass/mass-ratio quadrature converged at n = 150 | ≤ 3×10⁻⁴ nat |
| Likelihood cost | 0.45 ms/call vs ~40 ms for the TOV sweep |
| End-to-end mock run | ln B = −2.69 ± 0.27 against hadronic-truth data |
| End-to-end **real-data** run (FSU2L80, nlive 150) | ln Z = −28.58 ± 0.22, ln L_had = −25.54, **ln B = −3.03** |

The real-data number is a pipeline demonstration, not a production result —
nlive = 150 is too low for six dimensions with hard boundaries, and the open
issues in §4 all move it. Its sign is what you would expect: FSU2L80 already
satisfies the pulsar and tidal constraints reasonably well, so the hybrid
extension pays an Occam penalty without buying fit. Part of that penalty is the
unidentifiable second transition of §4.0.

### The Maxwell construction is genuine (your open item)

The polytropes are written in energy density, so they fix P(ε) but say nothing
about n. n is now reconstructed from the first law, dn/n = dε/(ε+P), integrated
outward from the table's own n column, and across each jump the density on the
far side is **forced** by μ continuity:

```
n₊ = n₋ (ε₋ + Δε + P)/(ε₋ + P)
```

Since Δε ≥ 0 and P > 0 this gives Δn ≥ 0 automatically. So "Maxwell
construction" is justified in print, not an overstatement. `HybridEOS.maxwell_report()`
prints the numbers a referee will ask for; on the test case:

```
jump 1: n = 2.73 → 3.56 n₀ (Δn/n = 0.304), μ = 1184.9 MeV, Δε/ε = 0.350
jump 2: n = 5.04 → 5.62 n₀ (Δn/n = 0.116), μ = 1285.8 MeV, Δε/ε = 0.133
```

One caveat, reported rather than hidden: the supplied tables' **outer crust** is
not exactly thermodynamically consistent — μ wanders non-monotonically by up to
5.5 MeV out of 938 over 17 steps below n ≈ 10⁻³ fm⁻³. Those layers carry no
mass, radius or tidal signal, but if you quote n anywhere, quote it above the
crust.

---

## 4. Open physics issues — these are your decisions, not bugs

### 4.0 The second transition is almost never inside a star — this is the big one

Over the approved prior, I counted how often any **stable** star reaches each
transition pressure (1200 draws, FSU2L80):

```
accepted by the hard constraints                 25.6%
  ... of those, a stable star reaches P_t         64.8%
  ... of those, a stable star reaches P_1          2.2%
```

And in the mock posterior (80 predictive draws): **0%**. The posterior on γ₂ and
Δε₁ came back indistinguishable from their priors — γ₂ median 4.82, 68%
[1.87, 8.47] against a U[0.5, 10] prior whose median is 5.25 and 68% interval is
[2.0, 8.5].

So for 97.8% of its own prior the "two-transition" model **is a one-transition
model** with two unidentifiable extra parameters. Those parameters do not sit
idle: they contribute a pure Occam penalty to ln Z_hybrid, so the headline Bayes
factor is biased *against* the hybrid model by an amount that has nothing to do
with the data.

Tightening the prior helps, but not for free:

| prior variant | accepted | reaches P_t | reaches P_1 | multi-branch | median M_TOV |
|---|---|---|---|---|---|
| as approved (ε_t ≤ 6 n₀, cap 1800) | 25.6% | 64.8% | 2.2% | 2.6% | 1.93 |
| ε_t ≤ 4 n₀, cap 1800 | 27.4% | 44.9% | 4.9% | 4.9% | 1.80 |
| ε_t ≤ 4 n₀, cap 1200 | 17.9% | 54.0% | 11.8% | 6.2% | 1.66 |
| ε_t ≤ 3 n₀, cap 1100, width ≤ 600 | 15.6% | 41.4% | 15.7% | 10.0% | 1.45 |

The reachability and the twin fraction both go up by ~6×, but the median M_TOV
falls from 1.93 to 1.45. **In this EOS family, an EOS whose star reaches the
second transition is nearly always too soft to support 2 M_⊙.** That is a
physical statement about energy-density polytropes with Δε ≥ 0, not a numerical
artefact: each polytrope restarts at low c_s² after a jump, so a second jump
deep enough to be inside the star costs the pressure support needed for M_TOV.

This reframes the result rather than killing it. Three honest ways forward:

1. **Report the one-transition model as the primary result** and the second
   transition as a nested extension, with its own internal Bayes factor. This is
   what the data can actually constrain, and it makes the Occam penalty a
   measured quantity rather than a hidden bias.
2. **Reparameterise so the second transition is forced inside the star** — e.g.
   sample ε₁ as a fraction of the central density of the maximum-mass
   configuration rather than on an absolute scale. This is a different prior, so
   it must be pre-registered, and it makes the prior EOS-dependent (defensible,
   but say so).
3. **Keep the prior and report the reachability fractions** in the paper, as the
   tables above. `analysis.py` now prints them for every run.

Whichever you pick, decide it before computing a real Bayes factor. This is the
forking path most likely to be caught in review.

### 4.1 The mass prior π(m|θ) is doing real work and is not yet justified

Every pulsar term carries a factor 1/|U|, where |U| is the measure of the EOS's
stable mass support above m_min = 1. This is an **Occam penalty on large M_TOV**,
and it is not small: it is why NL3 (M_TOV = 2.77) scores worse on J0740 (−3.20)
than FSUGarnet (M_TOV = 2.07, −1.85) even though NL3 comfortably accommodates a
2.07 M_⊙ star. It fights directly against the J0952 term, which rewards large
M_TOV.

Flat-in-m is the conventional choice and it is what is implemented, but it is a
*population* prior and it is not obviously right for objects selected by radio
timing. The alternatives in the skill (a fixed population prior, a
quadratic-in-central-density prior) give materially different Bayes factors.
**Decide this and state it**; it is more consequential than the branch rule.

Related inconsistency, deliberately left as you specified it: the J0952 term uses
the one-sided CDF Φ((M_TOV − μ)/σ), which corresponds to an *improper* flat prior
on m and therefore carries no 1/|U| factor, while the NICER terms do. The two
terms are not on the same footing. `LIKELIHOOD["mass_floor_mode"] = "marginal"`
makes them consistent; it moves the total by 0.025 nat on the baseline, so this
is a tidiness issue rather than a numerical one — but a referee will spot it.

### 4.2 The branch rules are different population priors, and "max" is not one

Writing the joint prior over (mass, branch) makes this unambiguous:

* **mixture** — pick m uniformly on the union of the branches' support, then
  uniformly among the branches present at that m. Weight 1/N_b(m), normalised
  by |U|. Your approved default.
* **sum** — pick (m, branch) uniformly on the disjoint union. Weight 1,
  normalised by S = Σ_b (branch span). Not double counting; the larger
  normaliser pays for it. It is a *different* prior, with an extra penalty on
  EOSs that have a lot of stable mass range.
* **max** — a profile over branches, not a marginal. It is not a likelihood.
  Keep it for comparison with Grundler & Li, label it as what it is.

Both legitimate rules are now correctly normalised (they were not before). On
the single-branch hadronic baseline all three agree exactly, as they must.

### 4.3 The polytrope form limits what you can claim

With P = Kε^γ, c_s² = γP/ε = γKε^(γ−1). At the base of polytrope 1 that is
γ₁P_t/(ε_t + Δε), which a large jump makes *small*; causality then bites only at
the top. So each segment is forced to run from soft to stiff, and the family
cannot reproduce constant-c_s² (CSS) models. Measured consequence: 57% of prior
draws are rejected for acausality (21% at the base of polytrope 1, 36% only
after the TOV solve), overwhelmingly from the γ ≤ 10 ceiling.

This is in your notes already. The thing to add in print is that the posterior is
**not comparable to CSS-parameterised results** such as Grundler & Li, and that
the γ ceiling is a prior choice that moves the evidence.

### 4.4 Masquerade, and what ln B can mean

M–R–Λ cannot identify microphysics. A positive ln B says the data prefer a stiff
segment above a density discontinuity; hyperons, Δ-isobars and kaon condensates
produce the same shape. This has to appear in the abstract, not a caveats
paragraph.

Related, and sharper: **Δε → 0 does not recover the hadronic model.** It leaves a
polytrope that still differs from the table. The hadronic model is recovered only
when ε_trans exceeds every stable central density — a prior *edge*, not a nested
point. So a Savage–Dickey ratio at Δε_trans = 0 tests "jump vs continuous-P kink",
which is a legitimate internal result, but it is **not** a check on ln B(hybrid vs
hadronic). Do not let the two get conflated.

### 4.5 Still to do before any real Bayes factor

1. **Freeze the source and scenario list before computing anything.** The J0030
   baseline is not chosen (PDT-U is wired in, Vinciguerra+2024 is the
   alternative), J0614's role is undecided, GW190425 is untested. Forking paths
   here are the single biggest threat to the result's credibility.
2. The encompassing-prior Bayes factor for γ₁ < γ₂ needs its prior odds computed
   by Monte Carlo *through the causality filter*, which truncates the prior
   non-uniformly in γ. Rectangle geometry gives the wrong answer.
3. Injection–recovery, including a null pure-hadronic injection.
4. PIDONet cannot yet stand in for the solver: the generator makes smooth c_s²
   with no Maxwell jumps and no twin branches, while the inference prior has
   both. Train on hybrid-prior draws before cross-validating.

---

## 5. Files changed

| file | change |
|---|---|
| `solver.py` | branch detection rewritten (`_dM_sign`, `_mass_noise`, `_thin_grid`, stopping rule in `_extrema_insert`) |
| `hybrid_eos.py` | c_s² envelope; `number_density()` and `maxwell_report()`; `n_of_eps`, `mu_of_eps` |
| `likelihood.py` | rewritten: `MassRadiusTerm`, `GWTerm`, `MassFloorTerm`, correct normalisers |
| `datasets.py` | **new** — provenance-checked loaders, weighted FFT KDE, headline tests, sampling-prior forms |
| `config.py` | real data paths with md5/DOI/column maps; `eps1_cap`; `min_dlnPc`, `noise_factor`; bandwidth and population-prior settings |
| `priors.py` | ε₁ cap and its truncation |
| `mockdata.py` | builds likelihood *terms*, so mock runs use the real code path |
| `validate.py` | 9 test groups, pass/fail, causal test case, exits non-zero on failure |
| `run_inference.py` | new likelihood API; provenance and per-term breakdown in `summary.json` |
