# Pre-registration: frozen analysis configuration

**Frozen 2026-11-01, by the project owner, before any Bayes factor was computed
under the one-transition model.**

The purpose of this file is to remove the forking path. Every number reported as
a *primary* result comes from exactly the configuration below. Anything else is a
sensitivity analysis and must be labelled as one, whether or not it is more
favourable. If a frozen choice is changed after a primary number has been seen,
that is a new analysis and this file gets a new dated section saying so.

## 1. Model

One hadronic table, one Maxwell first-order transition, one energy-density
polytrope running to the centre of the heaviest stable star.

    theta = (eps_trans, delta_eps_trans, gamma),  ndim = 3

Priors (`config.PRIOR`): `eps_trans` uniform in [1, 6] n0 via the table's own
n(eps); `delta_eps_trans / eps_trans` uniform on [0, 1.5]; `gamma` uniform on
[0.5, 10]. Hard constraints and the surviving prior fraction: see
`model.screen()` and `acceptance.py`.

Baseline: the pure hadronic table, no free parameters, so Z_had = L_had and
ln B = ln Z_hybrid - ln L_had.

## 2. Sources and scenarios (frozen)

| source | release | identifier | md5 |
|---|---|---|---|
| PSR J0030+0451 | Kini et al. 2026, **PDT-U** | arXiv:2602.23743, Zenodo 10.5281/zenodo.18741941 | `e2d50fdca6e5a4a2ac447d2e6afe1d26` |
| PSR J0437−4715 | Miller et al. 2026 | ApJL 1000 L48, arXiv:2512.08790, Zenodo 17833896 | `2e70f5f9a9bf35aaead616df581ee584` |
| PSR J0740+6620 | Salmi et al. 2024 | arXiv:2406.14466, Zenodo 10.5281/zenodo.10519473 | `f3470465f6a9bddd8f3cf46fd6975240` |
| GW170817 | **GWTC-1**, IMRPhenomPv2NRT low-spin | LIGO DCC P1800370, data DOI 10.7935/82H3-HH23 | `58e919868f0fae590aad2348592e7ea2` |
| PSR J0952−0607 | Romani et al. 2025, 2.35 +/- 0.11 Msun | arXiv:2512.05099 | (published value, not a file) |

Superseded alternatives, **not** used for primary numbers: Kini ST+PDT and
Vinciguerra+2024 for J0030; Choudhury+2024 for J0437; the PRX P1800061 and
high-spin releases for GW170817; the 0.17 Msun J0952 width from arXiv:2207.05124.

## 3. Likelihood choices (frozen)

- **Branch rule:** `mixture`. Justification: in the one-transition prior only
  2.2% of accepted draws have more than one stable branch, and on those the
  second branch is small enough that `mixture`, `max` and `sum` agree to
  0.01 nat. The rule is therefore not doing measurable work, and `mixture` is
  the one that is a proper marginal.
- **Mass prior:** flat in m over the EOS's own stable support, normalised, i.e.
  carrying the 1/|U| factor (Legred et al. 2021, arXiv:2106.05313).
  `mass_prior_min = 1.0`, which also matches the lower edge of the J0030
  sampling prior.
- **J0952 mass floor:** `mass_floor_mode = "marginal"` — the same integral
  against pi(m|theta) as the NICER terms, so every pulsar term uses one
  population prior.
- **GW population prior:** `uniform_mass`, matching the release's own sampling
  prior in component masses, so the q-shapes cancel inside the integral and
  survive in the normaliser.
- Bandwidths `bw_factor_mr = 0.3`, `bw_factor_gw = 1.0`; quadrature
  `n_mass_grid = 150`, `n_q_grid = 120`, `q_min = 0.2`.

## 4. Sampler (frozen)

`nlive = 500`, `dlogz = 0.1`, `sample = "auto"`, `bound = "multi"`, seed 1.
Evidence renormalised by the measured prior acceptance: `acceptance.py` with
`n_draws = 4000`, `seed = 7`, per hadronic table.

## 5. Sensitivity analyses declared in advance

These are pre-registered as secondary, to be reported alongside the primary
numbers whichever way they come out. Declaring them here is what makes them
sensitivity analyses rather than forks.

1. **Mass prior.** Measured 2026-11-01, over 120 accepted prior draws per
   table: a causal Maxwell transition cannot raise M_TOV above the hadronic
   table's own value (equality in the delta-eps -> 0 limit, and causality caps
   the polytrope), so |U|_hybrid <= |U|_hadronic always and the 1/|U| factor
   can only *help* the hybrid model, never penalise it. The contribution to
   ln B is +0.09 nat at the median for FSU2L80 (+0.05 for NL3) but reaches
   +4.2 nat (+4.6) in the 95th percentile, for draws that crush M_TOV to
   ~1.4 Msun. So the risk is the opposite of an Occam penalty: the normaliser
   hands a multi-nat bonus to exactly the heavily-softened EOSs this study
   would call a rescue, and the data (J0952 at 2.35, J0740 at 2.07) have to
   remove it. Report ln B under a theta-independent population mass prior, and
   under an unnormalised per-branch prior, for at least FSU2L80 and NL3, and
   state how much of the primary ln B is the normaliser.
2. **J0437.** Choudhury et al. 2024 as the co-primary release, since its radius
   posterior differs materially from Miller's bimodal one.
3. **J0030.** Kini ST+PDT.
4. **GW170817.** The PRX P1800061-v11 low-spin release.
5. **eps_trans floor.** A 3 n0 floor instead of 1 n0.
6. **Delta-eps parameterisation.** Absolute instead of fractional, since a
   prior reparameterisation moves the evidence.

## 5a. A limit of the model, worth stating in the paper

With causality enforced, a Maxwell transition can only soften: M_TOV of the
hybrid EOS is bounded above by M_TOV of the hadronic table it is built on
(measured over accepted prior draws; the bound is saturated as
delta_eps_trans -> 0). The rescue mechanism this model can express is therefore
*reducing radii and M_TOV*, not raising the maximum mass. A hadronic table that
is already too soft to satisfy the J0952 floor cannot be rescued by a transition
in this parameterisation, and a null result for such a table says nothing about
first-order transitions in general.

## 6. Not frozen, still open

- J0952 rotation (the published mass assumes a rotating configuration; the
  solver is non-rotating).
- Null injections: a hadronic-truth mock should return ln B <= 0, and the
  recovered ln B on mock data is the only calibration of the pipeline so far.
- Restricting eps_trans to values actually reached inside a stable star, which
  would change the prior volume rather than the likelihood.
