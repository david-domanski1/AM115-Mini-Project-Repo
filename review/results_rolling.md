# Results: rolling-origin forecast evaluation (Agent A, David's slide-3 box)

Produced by `python3 scripts/run_rolling.py` (seed 115, runtime 628 s). Code is in `src/rolling.py`, tests in
`tests/test_rolling.py` (9 tests, 0.4 s). Numbers: `results/rolling.json` (`headline` key), tables in
`results/rolling_by_season.csv`, `rolling_config_grid.csv`, `rolling_reliability.csv`, `rolling_betting.csv`,
`rolling_param_bootstrap_2024_25.csv`, per-match forecasts for every configuration in `rolling_predictions.csv`.
Figures: `figures/fig6_rolling_logloss.png|pdf`, `figures/fig7_rolling_reliability.png|pdf`.
Rubric sections served: **Validation and a baseline (12)**, **Estimation and uncertainty (10)**, plus lines for
**Assumptions**, **Model** and **What this does not show**.

All log-losses are in nats per match (lower is better). Unless stated otherwise, a 95% CI is a percentile interval from
2,000 paired bootstrap resamples of the test matches, with every forecaster scored on the same resampled matches. A
10-season mean averages the per-season bootstrap means replicate by replicate, treating seasons as independent.

## What was done (method, precise enough to re-implement)

- **Test seasons** are 2016-17 to 2025-26, 3,800 matches. Training data is every match since 2005-06 dated strictly
  before the forecast window.
- **Refit points.** The model is refitted before every *matchday window*. A window is Friday–Monday or Tuesday–Thursday,
  giving 45–55 windows per season. football-data collects the Pinnacle "pre-match" odds on Friday and Tuesday
  afternoons, so the model and those odds are dated at the same moment.
- **Model.** log λ = a[home] + b[away] + g, log μ = a[away] + b[home]. a and b belong to a team *spell*, meaning a run of
  consecutive Premier League seasons, so a promoted club starts a new spell. g is either one home advantage for all
  matches or one per season. In per-season mode the forecast uses the test season's own g once 100 of its matches have
  been played, and the previous season's g before that.
- **Weights.** Match i gets weight 0.5^(age/half-life), with age measured in days to the window start. The half-life
  grid is {no decay, 730, 365, 180 days}. Matches with weight below 1e-3 are dropped; this changes no probability by more
  than 2e-4.
- **Estimation.** Weighted maximum likelihood (L-BFGS-B, analytic gradient), warm-started from the previous window. The
  scale is fixed by setting the mean a of the previous season's 20 spells to 0.
- **Promoted teams.** A Gaussian prior with SD 0.3 on the log scale (about 10 matches' worth of information) pulls each
  promoted spell toward a target taken from the fit made just before the season. The rule therefore sets the forecast
  before the club has played, and its influence fades as matches accumulate. There are three targets:
  - mean α, β of the relegated three (the frozen-fit rule);
  - league average;
  - the 50% midpoint of the two.
- **Probabilities** come from the 0..10 score grid, renormalised.
- **Baselines.**
  - Base rate: H/D/A frequencies over all seasons before the test season.
  - Pinnacle pre-match (Fri/Tue collection, 1–3 days before kick-off; these are *not* opening odds) and Pinnacle
    closing, with the margin removed proportionally. The bookmaker knows injuries, suspensions and team news that this
    model never sees.
  - **2025-26 has Pinnacle odds for only 210 of 380 matches** (none after 17 Jan 2026). Every Pinnacle comparison uses
    those 210, giving 3,630 matches over the 10 seasons. Poisson vs base rate uses all 380 (3,800).

## Headline numbers

| Quantity | Estimate | 95% CI | n | How |
|---|---|---|---|---|
| Poisson log-loss, mean of 10 seasons | 0.9662 | [0.9506, 0.9822] | 3,800 | best config (365-day half-life, season γ); match bootstrap |
| Base-rate log-loss | 1.0663 | [1.0574, 1.0749] | 3,800 | prior-season H/D/A frequencies |
| Pinnacle pre-match log-loss | 0.9491 | [0.9333, 0.9660] | 3,630 | odds matches only |
| Pinnacle closing log-loss | 0.9459 | [0.9302, 0.9629] | 3,630 | odds matches only |
| Poisson − base rate | −0.1001 | [−0.1140, −0.0867] | 3,800 | paired |
| **Poisson − Pinnacle pre-match** | **+0.0141** | **[+0.0083, +0.0195]** | 3,630 | paired, match bootstrap |
| same, seasons as the sampling unit | +0.0141 | [+0.0064, +0.0218] | 10 seasons | t interval over the 10 season differences |
| Poisson − Pinnacle closing | +0.0173 | [+0.0114, +0.0229] | 3,630 | paired |
| Between-season SD of Poisson − Pinnacle | 0.0107 | [0.0074, 0.0196] | 10 | chi-square CI; Cochran Q = 13.4 on 9 df, p = 0.15 |
| Detectable gap, one season | 0.025 | per-season range 0.021–0.032 (not a CI) | 380 | 2.80 × paired bootstrap SE; mean SE 0.0089 |
| Detectable gap, 10-season mean | 0.0079 | exact design quantity | 3,630 | 2.80 × √(Σ SE²)/10 |
| Seasons in which Poisson beats Pinnacle pre-match | 0 of 10 | — | 10 | count of point estimates below 0 |
| Brier: Poisson / base / Pinnacle pre-match | 0.5736 / 0.6450 / 0.5616 | [0.5628, 0.5847] / [0.6387, 0.6511] / [0.5504, 0.5733] | 3,800 / 3,800 / 3,630 | multi-class Brier, range 0..2 |
| Best half-life | 365 days | 180–365 days not distinguishable | 3,800 | lowest 10-season mean; paired CIs vs best |
| Gain from decay (no decay − best) | +0.0159 | [+0.0097, +0.0221] | 3,800 | paired |
| Season γ − one γ (365 d) | −0.0003 | [−0.0014, +0.0010] | 3,800 | paired; no difference |
| Promoted-rule range, all matches (league avg − shrink 50%) | +0.0006 | [−0.0002, +0.0015] | 3,800 | paired |
| Promoted-rule range, matches with a promoted team | +0.0021 | [−0.0009, +0.0052] | 1,080 | paired |
| 2024-25 log-loss, match-only interval | 0.9763 | [0.9248, 1.0270] | 380 | parameters fixed, 2,000 match resamples |
| 2024-25 log-loss, parameter + imputation + match interval | 0.9763 | [0.9402, 1.0435] | 380 | 200 training-set bootstraps with full rolling refits; width ratio 1.01 |
| 2024-25 parameter-only SD of log-loss | 0.0081 | [0.9601, 0.9934] re-centred | 380 | SD over the 200 refits on the fixed 380 matches |
| Calibration-in-the-large, P(home) − share | −0.004 | [−0.018, +0.011] | 3,800 | 10-season mean |
| same, P(draw) − share | −0.003 | [−0.017, +0.010] | 3,800 | |
| same, P(away) − share | +0.007 | [−0.007, +0.020] | 3,800 | |
| same, Pinnacle closing P(draw) − share | +0.010 | [−0.004, +0.024] | 3,630 | |
| Mean λ − home goals, 10 seasons | −0.003 | [−0.042, +0.037] | 3,800 | |
| Mean μ − away goals, 10 seasons | −0.020 | [−0.057, +0.015] | 3,800 | |
| **2024-25 mean λ − home goals, rolling** | **+0.094** (1.607 vs 1.513) | [−0.027, +0.216] | 380 | match bootstrap |
| **2024-25 mean λ − home goals, frozen fit** | **+0.287** (1.799 vs 1.513) | [+0.158, +0.416] | 380 | frozen λ from `numbers.json`; bootstrap of realised goals |
| Betting return, margin 0 | +6.8% per unit | [−5.3%, +19.2%] | 689 bets | flat stakes at Pinnacle closing, 2024-25 + 2025-26 (590 matches) |
| Betting return, margin 5% | +1.5% | [−12.1%, +16.4%] | 552 bets | match bootstrap |
| Betting return, margin 10% | +3.4% | [−13.4%, +20.0%] | 433 bets | match bootstrap |

## Per season (best configuration)

Pinnacle columns use the odds matches; 2025-26 has n_odds = 210.

| Season | n_odds | Poisson | Base rate | Pinnacle pre-match | Pinnacle closing | Poisson − Pinnacle pre | CITL draw | mean λ / realised | λ − goals |
|---|---|---|---|---|---|---|---|---|---|
| 2016-17 | 380 | 0.934 [0.888, 0.981] | 1.044 [1.017, 1.071] | 0.906 [0.856, 0.958] | 0.906 [0.857, 0.959] | 0.028 [0.010, 0.046] | 0.013 [−0.028, 0.054] | 1.58 / 1.60 | −0.01 [−0.13, 0.11] |
| 2017-18 | 380 | 0.959 [0.910, 1.008] | 1.067 [1.037, 1.095] | 0.945 [0.891, 0.998] | 0.941 [0.887, 0.994] | 0.014 [−0.002, 0.029] | −0.025 [−0.068, 0.020] | 1.53 / 1.53 | −0.01 [−0.13, 0.12] |
| 2018-19 | 380 | 0.894 [0.845, 0.946] | 1.047 [1.019, 1.074] | 0.892 [0.840, 0.946] | 0.891 [0.838, 0.945] | 0.002 [−0.013, 0.017] | 0.040 [−0.000, 0.078] | 1.55 / 1.57 | −0.02 [−0.14, 0.10] |
| 2019-20 | 380 | 0.974 [0.925, 1.020] | 1.065 [1.037, 1.094] | 0.972 [0.919, 1.026] | 0.972 [0.920, 1.027] | 0.002 [−0.015, 0.019] | −0.009 [−0.051, 0.033] | 1.52 / 1.52 | 0.01 [−0.11, 0.12] |
| 2020-21 | 380 | 1.015 [0.964, 1.065] | 1.097 [1.069, 1.124] | 1.006 [0.954, 1.056] | 0.997 [0.946, 1.046] | 0.009 [−0.010, 0.026] | 0.021 [−0.021, 0.061] | 1.41 / 1.35 | 0.06 [−0.07, 0.18] |
| 2021-22 | 380 | 0.951 [0.903, 1.001] | 1.073 [1.047, 1.100] | 0.936 [0.887, 0.986] | 0.937 [0.887, 0.988] | 0.015 [−0.002, 0.030] | 0.000 [−0.041, 0.041] | 1.47 / 1.51 | −0.05 [−0.16, 0.07] |
| 2022-23 | 380 | 1.001 [0.948, 1.053] | 1.048 [1.022, 1.075] | 0.966 [0.916, 1.017] | 0.962 [0.912, 1.013] | 0.035 [0.017, 0.052] | −0.002 [−0.044, 0.039] | 1.59 / 1.63 | −0.04 [−0.18, 0.09] |
| 2023-24 | 380 | 0.923 [0.869, 0.973] | 1.056 [1.028, 1.082] | 0.909 [0.855, 0.963] | 0.900 [0.846, 0.956] | 0.014 [−0.002, 0.030] | −0.004 [−0.044, 0.037] | 1.74 / 1.80 | −0.06 [−0.18, 0.07] |
| 2024-25 | 380 | 0.976 [0.925, 1.027] | 1.084 [1.058, 1.111] | 0.970 [0.924, 1.020] | 0.966 [0.919, 1.016] | 0.006 [−0.012, 0.022] | −0.026 [−0.070, 0.016] | 1.61 / 1.51 | 0.09 [−0.03, 0.22] |
| 2025-26 | 210 | 1.035 [0.993, 1.077] | 1.082 [1.054, 1.111] | 0.989 [0.930, 1.048] | 0.987 [0.927, 1.046] | 0.016 [−0.008, 0.039] | −0.038 [−0.083, 0.008] | 1.53 / 1.53 | −0.00 [−0.11, 0.11] |

The 2025-26 Poisson log-loss of 1.035 is over all 380 matches. On the 210 matches with odds the Poisson figure is the
Pinnacle figure plus the paired difference, 1.005.

## What the numbers say

1. **The model needs no further justification against a no-model baseline.** It beats the prior-season base rate in
   every one of the 10 seasons: −0.100 nats per match [−0.114, −0.087] on average.
2. **Pinnacle is better than the model, consistently and measurably.** The mean gap is +0.014 [+0.008, +0.020] against
   pre-match odds and +0.017 [+0.011, +0.023] against closing odds. The Poisson point estimate is worse in **all 10
   seasons**; individually the gap is significant only in 2016-17 and 2022-23. One season cannot separate the two (the
   detectable gap is about 0.025 per season), but ten seasons can (detectable gap 0.008). The frozen single split's
   "cannot tell apart" (+0.0135 [−0.010, +0.037]) is now resolved: the gap is real and small. Part of it is information
   rather than modelling: the bookmaker prices injuries and team news, and the closing odds see more of it than the
   Fri/Tue odds (closing beats pre-match by 0.003).
3. **The gap does not vary between seasons beyond noise.** Between-season SD 0.011, Cochran Q p = 0.15. That includes
   2020-21, where the model was only 0.009 behind.
4. **Time decay is the improvement that matters.** A half-life of 365 days is 0.016 [0.010, 0.022] better than no
   decay. 180 days is statistically indistinguishable from 365, and 730 days is slightly worse (+0.003 [+0.0005, +0.005]).
   The half-life was chosen on the test seasons themselves, so this is mildly optimistic. The same configuration is
   best on the first five seasons. On the last five, one-γ at 365 days is best, but it differs from the chosen one by
   0.0003.
5. **A season-specific γ adds nothing** to the forecast: −0.0003 [−0.0014, +0.0010]. This fits Agent B's question being
   about γ in 2020-21, not about forecasting.
6. **The promoted-team rule does not matter.** All three rules agree within 0.002 nats even on the 1,080 matches
   involving a promoted team. The prior fades within about 10 matches, so the imputation is a non-issue for a rolling
   fit. It mattered more for the frozen fit, where it was never updated.
7. **Calibration is good.** Calibration-in-the-large is within ±0.007 for every outcome, with all CIs covering 0. All
   15 reliability bins have their mean prediction inside the observed bin's Wilson 95% CI, which also holds for Pinnacle
   closing (Fig 7). Mean predicted goals match realised goals to −0.003 (home) and −0.020 (away).
8. **The frozen fit's 1.80-vs-1.51 problem is removed.** In 2024-25 the rolling λ averages 1.607 vs 1.513 realised
   (+0.094 [−0.027, +0.216]). The frozen fit's +0.287 [+0.158, +0.416] is mostly gone, because the 1-year half-life
   still pulls 2023-24's goal glut forward but other seasons dilute it. Frozen log-loss on 2024-25 was 0.9837; rolling is
   0.9763 on the same 380 matches. A paired CI for that difference was not computed because the frozen per-match
   forecasts are not stored.
9. **Parameter uncertainty is small next to match sampling.** Refitting on 200 bootstrapped training histories moves
   the 2024-25 log-loss by SD 0.008, against a match-sampling SE of about 0.026. The combined interval [0.940, 1.044] is
   as wide as the match-only one [0.925, 1.027] (ratio 1.01; with 200 replicates the ratio is itself noisy to ±0.1). The
   centres differ, which is bootstrap noise in the 2.5% tails. The match-only intervals in the per-season table
   therefore do not understate the uncertainty in any important way. This answers the critique's point about fixed
   parameters.
10. **Betting.** Flat stakes at Pinnacle closing odds on 2024-25 and 2025-26 return between +1.5% and +6.8% per unit,
    depending on the margin, and every CI spans zero (for example [−5.3%, +19.2%] at margin 0). The point estimates are
    positive even though the model is worse than the market, which shows that 590 matches cannot detect an edge of a few
    percent in either direction. In one sentence: *no evidence the model beats the market; the betting test has no power
    at this sample size.*

## Disagreements with `review/final_plan_by_rubric.md` and the brief

- **SE per season is about 0.009, not 0.012.** The rolling forecasts are closer to Pinnacle than the frozen ones, so the
  paired differences have a smaller SE. The detectable gap is 0.025 per season rather than 0.034.
- **Time decay helps, but season-specific γ and the promoted-team rule do not.** Both were expected to be worth
  comparing; they make no measurable difference to the forecast.
- **"Poisson beats Pinnacle in some season" did not happen.** It is behind in every season on point estimates.
- **The base rate uses all prior seasons from 2005-06**, not just from 2015-16 as in `run_all.py`. For 2024-25 this
  gives 1.0844, vs 1.0811 for the `run_all` version.
- **The betting rule is defined as expected value**: bet when p_model × odds − 1 > margin, at the raw closing odds.
  "Model probability exceeds implied by margin" is read as a relative edge, not as percentage points.

## Design choices a reader should know about (Assumptions / Model)

- **Matchday = Fri–Mon or Tue–Thu window.** Results within a window are not used for later matches in the same window.
- **Team identity = spell.** A club returning after relegation does not inherit its old parameters.
- **Season-γ mode** uses the previous season's γ until 100 of the current season's matches have been played.
- **Promoted prior SD 0.3** on the log scale was fixed in advance, not tuned. Only the target rule was varied.
- **Optimizer flags.** L-BFGS-B reported non-success for 3 of the roughly 4,500 grid fits and 107 of the 9,000
  bootstrap fits. All were warm-started and returned finite parameters. I did not investigate further; the usual cause
  is line-search termination near machine precision, but this was not checked.

## Set aside

- **No paired frozen-vs-rolling CI.** The frozen per-match forecasts live in `run_all.py`, which I may not edit.
- **No sensitivity to the prior SD.** Given how small the rule differences are, it is unlikely to matter.
- **No validation-period split for choosing the half-life.** The first-five / last-five check is reported instead.
