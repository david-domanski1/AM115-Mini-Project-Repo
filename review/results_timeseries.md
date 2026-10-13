# Results: home advantage over time (workstream B)

Generated 2026-10-13T16:40:20 by `python3 scripts/run_timeseries.py` in 1013 s (seed 115; per-component seeds in `results/timeseries.json` → `meta.seeds`). Premier League 2005-06 to 2025-26, 21 seasons × 380 matches = 7,980 matches. Every number has a 95% interval and the way it was obtained. `results/timeseries.json` → `headline` has the machine-readable version of each.

Rubric sections served: **Verification (8)** in §1 and §5, **Estimation and uncertainty (10)** in §1, §2, §4 and §5, **Assumptions (8)** in §3 (crowd status equals season label; independent matches) and §4, and **What this does not show (4)** in §3 and §6.

## 1. γ per season and the goals-ratio identity (Verification, Model, Estimation)

**Analytic solution the code is checked against.** For one season of a double round robin with geometric-mean(α) = 1, the γ score equation gives Σ(x − λ) = 0, and summing the α and β score equations gives Σ(x − λ) + Σ(y − μ) = 0, so Σλ = ΣH and Σμ = ΣA. Every ordered pair (i, j) is played once, so Σ α_h β_a = Σ α_a β_h, and therefore

    γ̂ = Σ home goals / Σ away goals   (exactly),    extra home goals per match = mean(λ − λ/γ̂) = mean H − mean A.

The same argument with season-specific strengths and one shared γ gives the pooled common-γ MLE = ΣH/ΣA over the pooled seasons. The 40-parameter model adds nothing to the point estimate of γ. It matters for the forecasts and for the bootstrap: a resample is not a balanced schedule, so each replicate needs a refit.

- Identity check, L-BFGS-B γ̂ against ΣH/ΣA, all 21 seasons: max |difference| = **3.5e-07** (tolerance 1e-6). Pooled common-γ fits: 3.4e-07. Extra-goals identity: 3.8e-07. Bundesliga and La Liga seasons (18 and 20 teams): 1.7e-07. In 500 simulated seasons: 6.7e-07.
- Reproduction: with the same seeds, the first 500 of the 2,000 bootstrap replicates reproduce `run_all.py`'s 500-replicate intervals for 2015-16..2025-26 to 2.2e-16.
- Intervals: percentile, 2000 iid match-bootstrap refits per season (warm-started at the season fit; seed 115 + season index − 10).

| Season | γ̂ = ΣH/ΣA | 95% CI | extra home goals | 95% CI | Pearson dispersion |
|---|---|---|---|---|---|
| 2005-06 | 1.427 | [1.260, 1.623] | 0.437 | [0.287, 0.590] | 0.988 |
| 2006-07 | 1.456 | [1.297, 1.654] | 0.455 | [0.314, 0.601] | 0.889 |
| 2007-08 | 1.380 | [1.226, 1.547] | 0.421 | [0.266, 0.576] | 0.999 |
| 2008-09 | 1.298 | [1.147, 1.479] | 0.321 | [0.173, 0.476] | 1.008 |
| 2009-10 | 1.581 | [1.397, 1.780] | 0.624 | [0.458, 0.786] | 0.937 |
| 2010-11 | 1.383 | [1.233, 1.555] | 0.450 | [0.291, 0.603] | 0.918 |
| 2011-12 | 1.307 | [1.163, 1.486] | 0.374 | [0.208, 0.549] | 1.008 |
| 2012-13 | 1.257 | [1.123, 1.404] | 0.318 | [0.162, 0.468] | 0.969 |
| 2013-14 | 1.317 | [1.170, 1.496] | 0.379 | [0.217, 0.551] | 1.023 |
| 2014-15 | 1.349 | [1.200, 1.532] | 0.382 | [0.231, 0.541] | 0.968 |
| 2015-16 | 1.235 | [1.096, 1.395] | 0.284 | [0.126, 0.443] | 0.982 |
| 2016-17 | 1.328 | [1.180, 1.504] | 0.395 | [0.234, 0.566] | 0.941 |
| 2017-18 | 1.335 | [1.185, 1.512] | 0.384 | [0.227, 0.545] | 0.949 |
| 2018-19 | 1.252 | [1.115, 1.402] | 0.316 | [0.154, 0.473] | 0.911 |
| 2019-20 | 1.258 | [1.116, 1.433] | 0.311 | [0.151, 0.478] | 0.910 |
| 2020-21 | 1.008 | [0.884, 1.143] | 0.011 | [-0.171, 0.181] | 1.090 |
| 2021-22 | 1.159 | [1.033, 1.309] | 0.208 | [0.046, 0.376] | 0.964 |
| 2022-23 | 1.341 | [1.189, 1.520] | 0.416 | [0.246, 0.583] | 1.010 |
| 2023-24 | 1.217 | [1.095, 1.364] | 0.321 | [0.149, 0.507] | 0.880 |
| 2024-25 | 1.065 | [0.956, 1.187] | 0.092 | [-0.067, 0.255] | 0.902 |
| 2025-26 | 1.247 | [1.121, 1.407] | 0.303 | [0.158, 0.461] | 0.883 |

Pearson dispersion after the fit (X² over home and away goals / (760 − 40)) ranges 0.88 to 1.09 (pooled over 21 seasons: 0.958). Under the Poisson model a season falls in [0.899, 1.106] with 95% probability (χ²₇₂₀/720). Seasons outside that range: 2006-07, 2023-24, 2025-26, all *below* it, which means under-dispersion relative to the fit, not over-dispersion. The earlier figure of 0.88–1.09 over 11 seasons is replaced by these 21-season numbers. There is no sign of over-dispersion, which supports the Poisson variance assumption for the bootstrap and the tests below.

## 2. One γ for all seasons? Cochran's Q, likelihood ratio, DerSimonian–Laird (Estimation, Assumptions)

Inputs: y_s = log γ̂_s with the bootstrap SD of log γ̂ as its SE. Q is the inverse-variance Cochran statistic. LR is 2 × (Σ per-season log-likelihoods − pooled common-γ log-likelihood), the pooled fit having season-specific strengths and one γ. τ is the DerSimonian–Laird between-season SD of log γ, with a Q-profile CI. The trend is an inverse-variance weighted regression of log γ̂ on season start year.

| Seasons | k | Cochran Q (df, p) | LR (df, p) | DL τ [95% CI] | common γ (pooled MLE) [95% CI] | trend in log γ per season [95% CI] |
|---|---|---|---|---|---|---|
| all 21 (2005-06..2025-26) | 21 | 52.2 (20, 0.0001) | 51.2 (20, 0.0001) | 0.077 [0.045, 0.130] | 1.286 [1.254, 1.321] | -0.0110 [-0.0161, -0.0059] |
| 20 crowd seasons (excl. 2020-21) | 20 | 37.7 (19, 0.0064) | 35.3 (19, 0.0127) | 0.060 [0.023, 0.108] | 1.301 [1.270, 1.337] | -0.0099 [-0.0142, -0.0055] |
| 11 core (2015-16..2025-26) | 11 | 21.8 (10, 0.0163) | 22.0 (10, 0.0149) | 0.065 [0.016, 0.148] | 1.217 [1.174, 1.264] | -0.0097 [-0.0256, 0.0063] |
| 10 core crowd seasons (excl. 2020-21) | 10 | 12.8 (9, 0.1732) | 12.1 (9, 0.2090) | 0.039 [0.000, 0.114] | 1.239 [1.195, 1.288] | -0.0098 [-0.0220, 0.0023] |

Every pooled interval is a **common-γ interval**: it assumes one γ for all seasons in the set. It comes from 2000 season-stratified bootstrap refits, where pooled replicate b uses replicate b of every season.

**Reconciliation with `review/final_plan_by_rubric.md` §1 row 3, with the actual numbers.** Over the 11 core seasons the plan's sentence holds. Among the 10 core crowd seasons there is no evidence of between-season variation (Q = 12.8 on 9 df, p = 0.17; LR = 12.1, p = 0.21). Adding 2020-21 rejects a common γ (Q = 21.8 on 10 df, p = 0.016; LR = 22.0, p = 0.015; the LR reproduces the earlier 22.0 on 10 df). Alexia's Q of 14.3 (p = 0.11) used different SEs (500-replicate SD of γ̂ rather than 2,000-replicate SD of log γ̂); here it is 12.8. **Over all 21 seasons the sentence does not hold.** The 20 crowd seasons already reject a common γ (Q = 37.7 on 19 df, p = 0.0064; LR p = 0.013). Home advantage has declined since 2005-06: the slope is -0.0099 [-0.0142, -0.0055] in log γ per season, about -9% in γ per decade, with p = 8e-06. The residual heterogeneity around the trend is Q = 17.8 on 18 df (p = 0.47). Proposed wording for the write-up: *"Over 2015-16 to 2025-26 we find no evidence of between-season variation among the 10 crowd seasons, and adding 2020-21 rejects a common γ. Over 2005-06 to 2025-26 home advantage declines by about 9% per decade, so a single γ is rejected even without 2020-21."* The common γ over the 20 crowd seasons (1.301 [1.270, 1.337]) is therefore an average over a declining series, not a constant.

### 2020-21 and 2024-25 against the other seasons

Three reference models, all on log γ with bootstrap SEs. *Common γ*: z = (y_s − fixed-effect mean) / √(se_s² + se_mean²). *Random effects*: z = (y_s − DL mean) / √(se_s² + τ² + se_mean²), with τ estimated on the reference seasons. *Trend*: z = (y_s − linear-trend prediction for that season) / √(se_s² + var(prediction)). The reference set always excludes 2020-21 and the season being tested.

| Set | Season | γ̂ | z common γ (p) | z random effects (τ) | z vs trend (trend-predicted γ) | γ̂ / common γ [95% CI] |
|---|---|---|---|---|---|---|
| all 21 (2005-06..2025-26) | 2020-21 | 1.008 | -3.80 (0.0001) | -2.82 (0.060) | -3.02 (1.238) | 0.775 [0.679, 0.884] |
| all 21 (2005-06..2025-26) | 2024-25 | 1.065 | -3.60 (0.0003) | -3.03 (0.037) | -2.14 (1.219) | 0.809 [0.721, 0.908] |
| 11 core (2015-16..2025-26) | 2020-21 | 1.008 | -3.00 (0.0027) | -2.59 (0.039) | -2.99 (1.239) | 0.814 [0.712, 0.931] |
| 11 core (2015-16..2025-26) | 2024-25 | 1.065 | -2.79 (0.0053) | -2.79 (0.000) | -2.27 (1.238) | 0.845 [0.750, 0.951] |

**2024-25 needs addressing directly.** It played in full stadiums, and its γ̂ = 1.065 [0.956, 1.187] is nearly as low as 2020-21's 1.008 [0.884, 1.143]. Under every reference model both seasons sit more than 2 SDs below the others, and they are about equally unusual except against the trend, where 2024-25 is less extreme. Against all other crowd seasons the z values (2020-21 vs 2024-25) are -3.80 vs -3.60 under a common γ, -2.82 vs -3.03 under random effects, and -3.02 vs -2.14 against the trend. The earlier rough figures were −3.3/−3.0 and −2.5/−2.3; the 11-season rows above are their recomputation. A single low season is therefore not evidence of an empty-stadium effect: a season as low as 2020-21 happened with full crowds four years later. The crowd evidence has to come from within-season contrasts (§3) and other leagues (§6), and those are weak.

## 3. Crowd status by match date, Premier League 2019-20 to 2021-22 (Assumptions, What this does not show)

**Attendance is not in the football-data files.** Crowd status is a flag from the match date (and, for December 2020, the home club), not a head count:

- *closed*: from the 2019-20 restart on 17 June 2020 (92 matches) to the end of 2020-21 (350 matches), apart from the limited matches below;
- *limited* (30 matches): 2–15 December 2020 with the home club in a tier-2 area (London, Liverpool City Region, Brighton & Hove, Hampshire: Arsenal, Chelsea, Crystal Palace, Fulham, Tottenham, West Ham, Liverpool, Everton, Brighton, Southampton; up to 2,000 fans), and 18–23 May 2021 (last two rounds, all grounds, up to 10,000 fans). **The club list comes from the tier allocation of 2 December 2020 and was not verified against attendance records.** Crystal Palace and Fulham are added to the brief's list because they are London clubs under the same rule. A date-only sensitivity flag treats all 21 matches of 2–15 Dec as limited;
- *full*: everything else (2019-20 before the restart and all of 2021-22).

Fit: season-specific α and β, one γ per crowd status. Percentile CIs come from 2000 season-stratified bootstrap refits; differences use the paired replicates of the same fit.

| Fit | Status | n | γ [95% CI] | raw ΣH/ΣA |
|---|---|---|---|---|
| E0_main | full | 668 | 1.176 [1.076, 1.282] | 1.193 |
| E0_main | closed | 442 | 1.069 [0.957, 1.192] | 1.043 |
| E0_main | limited | 30 | 1.273 [0.941, 1.646] | 1.361 |
| E0_dec2020_date_only | full | 668 | 1.177 [1.076, 1.282] | 1.193 |
| E0_dec2020_date_only | closed | 431 | 1.072 [0.957, 1.193] | 1.044 |
| E0_dec2020_date_only | limited | 41 | 1.179 [0.916, 1.527] | 1.260 |
| E0_restart_split | full | 668 | 1.198 [1.093, 1.308] | 1.193 |
| E0_restart_split | closed_restart_2019_20 | 92 | 1.271 [1.055, 1.524] | 1.315 |
| E0_restart_split | closed | 350 | 0.989 [0.867, 1.126] | 0.981 |
| E0_restart_split | limited | 30 | 1.226 [0.904, 1.601] | 1.361 |
| E0_two_status | full | 698 | 1.183 [1.085, 1.285] | 1.200 |
| E0_two_status | closed | 442 | 1.065 [0.955, 1.184] | 1.043 |

| Fit | Difference | est [95% CI] | share of replicates < 0 |
|---|---|---|---|
| E0_main | closed_minus_full | -0.108 [-0.247, 0.040] | 0.925 |
| E0_main | limited_minus_full | +0.096 [-0.250, 0.482] | 0.284 |
| E0_dec2020_date_only | closed_minus_full | -0.105 [-0.244, 0.042] | 0.922 |
| E0_dec2020_date_only | limited_minus_full | +0.002 [-0.278, 0.356] | 0.489 |
| E0_restart_split | closed_restart_2019_20_minus_full | +0.074 [-0.140, 0.304] | 0.260 |
| E0_restart_split | closed_minus_full | -0.209 [-0.367, -0.033] | 0.990 |
| E0_restart_split | limited_minus_full | +0.028 [-0.308, 0.417] | 0.425 |
| E0_two_status | closed_minus_full | -0.118 [-0.249, 0.017] | 0.958 |

**Reading.** Closed doors minus full crowds: -0.108 [-0.247, 0.040]. As expected the interval is wide, and it includes 0. The 92 restart matches of 2019-20 do not show lower home advantage: restart − full = +0.074 [-0.140, 0.304]. The closed matches of 2020-21 do: -0.209 [-0.367, -0.033]. The limited-crowd γ rests on 30 matches and is uninformative. So the treatment 'crowd status = season label' breaks in both directions: 2019-20 contains 92 empty-stadium matches with ordinary home advantage, and the low value is specific to 2020-21 rather than to empty stadiums as such. That fits a 2020-21 season effect (which could be crowds, fixture congestion, or chance; see 2024-25 above) as well as it fits a crowd effect.

## 4. Date-block and team-cluster bootstraps (Estimation, Assumptions: independent matches)

Blocks: the season's 380 matches sorted by date and cut into 38 consecutive blocks of 10. The 38 blocks are resampled with replacement. Team cluster: the 20 home teams are resampled with replacement, each bringing all 19 of its home matches. Each scheme uses 2000 refits. The analytic Wald width uses SE(log γ̂) = √(1/ΣH + 1/ΣA).

| Season | iid | date blocks 38×10 | team cluster (home) | analytic Wald |
|---|---|---|---|---|
| 2020-21 | 0.260: [0.884, 1.143] | 0.219: [0.901, 1.120] | 0.336: [0.836, 1.173] | 0.248 |
| 2024-25 | 0.231: [0.956, 1.187] | 0.300: [0.922, 1.223] | 0.304: [0.924, 1.228] | 0.251 |

Entries are the 95% interval width followed by the interval. The Monte Carlo error of a 2,000-replicate width is about ±3%. Results, as they came out: in 2020-21 the date-block interval is *narrower* than the iid one (0.84×). In 2024-25 it is wider (1.30×). The team-cluster interval is wider in both seasons (1.29× and 1.32×). With only 38 blocks or 20 clusters these widths are themselves noisy, and a team-cluster resample perturbs the balance of the schedule more than an iid resample does. Read this as: *the iid interval could be understated by up to about 32% if home-team clustering matters, but there is no consistent sign of within-season time dependence.* None of the schemes moves 2020-21's or 2024-25's interval enough to change a conclusion above. The 2020-21 interval contains 1 under every scheme.

## 5. CI coverage, cold start, and fit diagnostics (Verification, Estimation)

Truth = the 2023-24 fit (γ = 1.2171). 500 simulated seasons were drawn on the real fixture list (seed 615). Each was fitted, then refitted on 200 iid bootstrap resamples. Coverage is the share of 95% intervals that contain the true γ, with a Clopper–Pearson CI.

| Interval | coverage [95% CI] |
|---|---|
| bootstrap percentile (the one used in Fig 2) | 0.948 [0.925, 0.966] (474/500) |
| Wald on log γ, bootstrap SE | 0.950 [0.927, 0.967] (475/500) |
| analytic log(ΣH/ΣA) ± 1.96 √(1/ΣH + 1/ΣA) | 0.962 [0.941, 0.977] (481/500) |

Nominal 95% is inside every coverage CI, so the bootstrap error bars in Fig 2 have the coverage they claim when the model is true. Mean bootstrap SE(log γ̂) = 0.0571, mean analytic SE = 0.0569, and the SD of log γ̂ across simulations is 0.0575. Coverage under a misspecified data-generating process (over-dispersion, correlated goals, drifting strengths) is not tested.

**Cold vs warm start** (200 simulated 2023-24 seasons, seed 715, each fitted from the truth and from the default x0): γ̂ bias warm 0.0021 [-0.0071, 0.0113], cold 0.0021 [-0.0071, 0.0113] (bias ± 1.96 Monte Carlo SE). Max |γ̂_cold − γ̂_warm| = 6.2e-07, and max |θ_cold − θ_warm| = 2.5e-06. The warm start at the truth does not drive the recovery result, because the likelihood is concave and both starts reach the same optimum. Mean iterations: 27 warm, 29 cold.

**Bounds and gradients.** Over all 170,937 fits in this script (data fits, every bootstrap refit, every simulation): max |θ| = 3.24 against bounds of ±5. **No parameter came within 1.76 of a bound**, which is more than the 1 required. Max projected-gradient norm (∞-norm, in nats per unit log-parameter) = 1.9e-02. Fits not reporting convergence: 0. The per-season data fits reach at most 1.1e-04, and their bootstrap refits at most 6.7e-04. The largest values come from the 400–840-parameter pooled bootstrap refits, which stop on L-BFGS-B's relative-ftol criterion (1e-13). In a separate check, 60 such 21-season refits had gradients up to 8e-3; polishing them with two further warm-started runs changed γ̂ by at most 9e-7, so the residual gradient has no effect on any reported digit.

## 6. Bundesliga and La Liga, 2019-20 to 2021-22 (stretch item: a parallel comparison, not a pooled analysis)

Same fit as §3 with two statuses. *closed* = from the restart (Bundesliga 16 May 2020, La Liga 11 June 2020) to the end of 2020-21. *fans allowed* = everything else. The partial crowds in those leagues are **not** separated, because the dates were not verified: Bundesliga September–October 2020, capacity caps and renewed closures in 2021-22, and a few La Liga matches at the end of 2020-21. The 'fans allowed' group is therefore a mix, and the contrast is biased toward 0.

| League | γ fans allowed [95% CI] (n) | γ closed [95% CI] (n) | closed − fans allowed [95% CI] |
|---|---|---|---|
| Premier League (limited merged into fans allowed) | 1.183 [1.085, 1.285] (698) | 1.065 [0.955, 1.184] (442) | -0.118 [-0.249, 0.017] |
| Bundesliga (D1) | 1.235 [1.125, 1.354] (530) | 1.144 [1.025, 1.268] (388) | -0.091 [-0.247, 0.059] |
| La Liga (SP1) | 1.380 [1.265, 1.502] (650) | 1.190 [1.081, 1.316] (490) | -0.190 [-0.339, -0.031] |

All three point estimates are negative, and only La Liga's interval excludes 0. Per-season γ̂ (= ΣH/ΣA, analytic Wald CI) for each league is in `timeseries.json` → `crowd.other_leagues.<league>.per_season`. The data identify a difference between periods, not its cause.

## 7. Figures

- `figures/fig2_home_advantage_2005_2026.png|pdf` replaces fig2 and fig2b, which `run_all.py` still writes, so the integrator retires those. It shows γ̂ per season with 2,000-replicate CIs; the common-γ band (20 seasons excluding 2020-21, labelled as a common-γ interval); the random-effects 95% range for one season's γ; and the identity in the title. Extra home goals = mean H − mean A is in `results/timeseries_by_season.csv`.
- `figures/fig8_crowd_gamma.png|pdf`: γ by crowd status for the Premier League, Bundesliga and La Liga, and the closed − fans-allowed differences.

## 8. Where this disagrees with `review/final_plan_by_rubric.md`, and what was set aside

- §1 row 3: the plan says there is no evidence of between-season variation among the crowd seasons. That holds over the 10 core crowd seasons only. Over the 20 crowd seasons since 2005-06 a common γ is rejected (p = 0.006), because of a downward trend.
- The plan expected the block bootstrap to be wider. It is narrower for 2020-21 and wider for 2024-25. The team-cluster bootstrap is wider in both seasons.
- Replicates were *increased*: 2,000 bootstrap replicates per season instead of 500, and 500 × 200 coverage simulations instead of 200 × 200. The script still runs in 16.9 min.
- The plan's Dec 2020 tier-2 list is extended by Crystal Palace and Fulham (London clubs). The list is not verified against attendance records, and a date-only sensitivity fit is reported.
- No changes were made to `src/data.py` or `data/team_names.csv`. The other leagues are read through the existing `load_raw_season(label, league)` in `src/crowd.py`. Their team names stay in the football-data short form and are not added to the E0 name table.
- Set aside: coverage under a misspecified data-generating process; a full random-effects or time-series model for γ (the DL τ and the linear trend are reported instead); verifying the partial-crowd dates for the Bundesliga and La Liga.
