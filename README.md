# Home advantage and Poisson forecasting of Premier League matches

AM115 (Harvard, mathematical modelling) mini-project. The report is
`notebooks/final_report.ipynb`, in the rubric's order, executed from a clean kernel. Everything
is reproducible from a fresh clone (the raw CSVs are committed under `data/raw/`, so no internet
connection is needed):

```bash
pip install -r requirements.txt nbformat nbconvert jupyter ipykernel
make workstreams  # the three workstream scripts: about 7 + 10 + 1 minutes on a laptop
make all          # = python3 run_all.py (25 s)  followed by  python3 -m pytest -q tests (2 s)
make notebook     # execute notebooks/final_report.ipynb in place from a clean kernel (10 s)
```

`make all` alone reproduces the core pipeline; `results/SUMMARY.md` then says which workstream
result files are missing. Python 3.9+, numpy / pandas / scipy / matplotlib only (plus nbconvert
for the notebook).

## The question

How many extra goals per match does playing at home give a Premier League team, and was that
home advantage lower in 2020-21, when matches were played behind closed doors, than in the
other seasons? And does the model that produces the answer forecast better than a forecast
that uses no model, and how far is it from the betting market?

## The model

For a match between home team *h* and away team *a*:

    X ~ Poisson(λ),  Y ~ Poisson(μ),  independent
    λ = α_h · β_a · γ        μ = α_a · β_h

α = attack strength, β = defensive weakness, γ = common home-advantage multiplier.
Parameters are fitted by maximum likelihood on the log scale with `scipy.optimize.minimize`
(L-BFGS-B) and an analytic gradient. Outcome probabilities P(H), P(D), P(A) come from summing
the independent-Poisson score grid over 0..10 goals each side and renormalising; the Skellam
closed form for the goal difference is implemented as a cross-check.

**Identifiability.** The likelihood is unchanged by α → cα, β → β/c and by nothing else, so the
model has exactly one non-identifiable direction. The fit imposes a single constraint,
geometric mean of α = 1, leaving **40 free parameters per season**. The brief also suggested
fixing the geometric mean of β to 1; that second constraint removes a degree of freedom the
model actually has (it forces the geometric mean of the away rate μ to be exactly 1) and pushes
the league's away-scoring level into γ: for 2023-24 it gives γ = 1.64 instead of 1.22 with a
log-likelihood 24 nats worse. It is still computed for every season and stored under
`diagnostics.two_constraint_fit` in `results/numbers.json`.

**Identity (the verification check).** For a double round robin with that constraint the score
equations give γ̂ = Σ home goals / Σ away goals exactly, and extra home goals per match
= mean home goals − mean away goals. The fitted γ̂ matches the ratio to 3.5e-7 in all 21 seasons.

**Dixon–Coles.** P(x, y) = τ(x, y; λ, μ, ρ) Pois(x; λ) Pois(y; μ) with τ = 1 − λμρ (0-0),
1 + λρ (0-1), 1 + μρ (1-0), 1 − ρ (1-1), 1 otherwise; ρ fitted jointly, sign free
(`src/dixon_coles.py`).

## Data

Source: [football-data.co.uk](https://www.football-data.co.uk), file
`mmz4281/{SS}/{league}.csv`. Premier League (`E0`) seasons 2005-06 through 2025-26 (21 × 380
matches); Bundesliga (`D1`) and La Liga (`SP1`) 2019-20 through 2021-22 for the crowd-status
comparison. Columns used: `Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR`, Bet365 odds `B365H/D/A`,
Pinnacle pre-match `PSH/D/A` (from 2012-13), Pinnacle closing `PSCH/D/A`, and the market
average (`BbAvH/D/A` before 2019-20, `AvgH/D/A` after). The non-closing odds are **not opening
odds**: football-data's notes say they are collected on Friday afternoons (weekend games) and
Tuesday afternoons (midweek games), 1-3 days before kick-off, so the bookmaker already knows
injuries and selection news that a model fitted earlier does not. `data/team_names.csv` is the
raw → canonical team-name table. Implied probabilities use proportional margin removal,
p_k = (1/o_k) / Σ_j (1/o_j); the overround Σ_j 1/o_j − 1 is stored per match.

The core analysis (`run_all.py`) uses 2015-16 to 2025-26 (11 seasons, 4,180 matches); the
home-advantage time series uses all 21 seasons; the rolling forecast trains on everything
since 2005-06 and tests on 2016-17 to 2025-26. Pinnacle odds are absent for 170 of the 380
matches of 2025-26 (none after 17 January 2026); Pinnacle statistics for that season use the
210 matches that have them, and every such number says so.

## Results in brief (every number: estimate, 95% interval, how obtained; details in `results/SUMMARY.md` and `review/results_*.md`)

- Common γ over the 20 crowd seasons 2005-06..2025-26 (excluding 2020-21): 1.301 [1.270, 1.337],
  i.e. 0.360 [0.326, 0.396] extra home goals per match (2,000 season-stratified bootstrap refits).
  Home advantage declines by −0.0099 [−0.0142, −0.0055] in log γ per season, about 9% per decade,
  so a single γ is rejected over the 21 seasons even without 2020-21 (Cochran Q p = 0.006); among
  the 10 crowd seasons of 2015-16..2025-26 there is no evidence of variation (Q p = 0.17), and
  adding 2020-21 rejects a common γ there (p = 0.016).
- 2020-21 (closed doors): γ = 1.008 [0.884, 1.143], 0.011 [−0.171, 0.181] extra goals.
  2024-25 (full stadiums): γ = 1.065 [0.956, 1.187]. Under random effects their z-scores are
  −2.8 and −3.0, so a single low season is not evidence of a crowd effect. Closed doors minus full
  crowds within 2019-20..2021-22, flagged by date (attendance is not in the data): −0.108
  [−0.247, +0.040]; the 92 restart matches of 2019-20 show ordinary home advantage.
- Rolling-origin forecast, 10 test seasons: Poisson 0.966 [0.951, 0.982] nats per match vs
  base rate 1.066 [1.057, 1.075] (Poisson − base −0.100 [−0.114, −0.087], better in all 10
  seasons) vs Pinnacle pre-match 0.949 [0.933, 0.966] (Poisson − Pinnacle +0.014 [+0.008, +0.020],
  behind in all 10 seasons; detectable gap 0.025 in one season, 0.008 over ten). A 365-day
  half-life is worth 0.016 [0.010, 0.022] over no decay; season-specific γ and the promoted-team
  rule make no measurable difference. Calibration-in-the-large within ±0.007 for every outcome.
- Dixon–Coles: ρ = −0.051 [−0.097, −0.009] (LR p = 0.022); it moves held-out log-loss by
  −0.0008 [−0.0018, +0.0002] nats per match. The pooled scoreline table has four of 17 cells
  beyond 2 Pearson SDs (2-2 at +3.3), outside the cells τ touches; home goals are mildly
  under-dispersed (0.948, Poisson range [0.969, 1.031]).
- Verification: grid vs Skellam 9e-16; γ̂ = ΣH/ΣA to 3.5e-7; 95% bootstrap intervals cover the
  true γ in 0.948 [0.925, 0.966] of 500 simulated seasons; cold and warm starts agree to 6e-7;
  no parameter within 1.76 of the ±5 bounds over 170,937 fits.

## Layout

| Path | Contents |
|---|---|
| `notebooks/final_report.ipynb` | **the report**, rubric order, every cell calls `src/`; `notebooks/exploration.ipynb` is the older thin notebook |
| `run_all.py` / `Makefile` | core pipeline driver (`make all`), workstream targets (`make workstreams`, `make notebook`) |
| `scripts/run_timeseries.py` | home advantage 2005-06..2025-26, common-γ tests, crowd flag (E0, D1, SP1), block bootstraps, CI coverage, cold start, fit diagnostics (7 min) |
| `scripts/run_rolling.py` | rolling-origin forecast evaluation 2016-17..2025-26, calibration, parameter-uncertainty bootstrap, betting test (10 min) |
| `scripts/run_dixon_coles.py` | Dixon–Coles ρ per season and pooled, scoreline table with residuals, histogram envelope, forecast effect (1 min) |
| `src/data.py` | download (cached), clean, team-name table, implied probabilities; `league` argument for D1/SP1 |
| `src/model.py` | log-likelihood, gradient, constrained parametrisation, fit, score grid, Skellam |
| `src/bootstrap.py`, `src/simulate.py`, `src/evaluate.py` | weighted-refit bootstrap, synthetic seasons and recovery, log-loss / Brier |
| `src/timeseries.py`, `src/crowd.py` | workstream B functions (Q, DerSimonian–Laird, trend, block/cluster weights, coverage, crowd-status fits, fig2, fig8) |
| `src/rolling.py` | workstream A functions (matchday windows, spells, decay weights, penalised fit, paired bootstrap, reliability, betting, fig6, fig7) |
| `src/dixon_coles.py` | workstream C functions (τ, joint MLE, profile likelihood, bootstrap, scoreline table, envelope, fig3, fig9) |
| `src/report.py` | notebook helpers: load results, render `headline` dicts, live checks, sections 7–8 prose from the results files |
| `src/figures.py` | style, `save`, fig1, fig2 (11 seasons), fig4, fig5 |
| `tests/` | pytest: Skellam vs grid, gradient, recovery, data invariants, and one file per workstream (49 tests, 2 s) |
| `results/numbers.json`, `results/SUMMARY.md` | every number of the core pipeline; plain-English summary incl. the workstream headlines |
| `results/{timeseries,rolling,dixon_coles}.json` | workstream results, each with `meta` (seed, runtime) and `headline: {est, ci, n, how}` |
| `review/` | the work plan, the rubric-mapped critique, and one results note per workstream |

## Where each figure comes from

| Output | Produced by | Contents |
|---|---|---|
| `figures/fig1_baselines` | `figures.fig1_baselines` | home / draw / away share by season, 2020-21 marked |
| `figures/fig2_home_advantage_2005_2026` | `timeseries.fig2_home_advantage_2005_2026` | **the home-advantage figure**: γ̂ by season 2005-06..2025-26, 2,000-replicate CIs, common-γ band (20 crowd seasons), random-effects range; identity in the title |
| `figures/fig2_home_advantage` | `figures.fig2_home_advantage(kind="goals")` | the same 11 core seasons in extra home goals (run_all.py) |
| `figures/fig3_goals_envelope` | `dixon_coles.fig3_goals_envelope` | pooled goals-per-match histograms vs fitted Poisson with a parametric-bootstrap band |
| `figures/fig4_parameter_recovery` | `figures.fig4_parameter_recovery` | fitted vs true log α, log β over 200 simulated seasons; histogram of γ̂ |
| `figures/fig5_logloss_comparison` | `figures.fig5_logloss_comparison` | single-split 2024-25 log-loss: base rate, Poisson, Pinnacle, paired differences |
| `figures/fig6_rolling_logloss` | `rolling.fig6_rolling_logloss` | rolling evaluation: log-loss per test season, Poisson − Pinnacle per season |
| `figures/fig7_rolling_reliability` | `rolling.fig7_rolling_reliability` | reliability diagram, Poisson vs Pinnacle closing |
| `figures/fig8_crowd_gamma` | `timeseries.fig8_crowd_status` | γ by crowd status, three leagues, and the closed-minus-fans differences |
| `figures/fig9_dixon_coles` | `dixon_coles.fig9_dixon_coles` | ρ by season, pooled profile likelihood, scoreline Pearson residuals |

`results/numbers.json` keys: `baselines`, `poisson_fit_by_season`, `pooled_gamma`,
`pooled_gamma_excluding_2020_21`, `covid_comparison`, `scorelines_2023_24`,
`team_strengths_2023_24`, `verification`, `preliminary`, `legacy` (the pre-13-October numbers
for the before/after table), `diagnostics`, `meta`.

## Conventions

- Seasons are labelled `2015-16` everywhere. Log-loss is always in nats per match. The Brier
  score is the multi-class sum over H/D/A (range 0 to 2).
- Bootstrap CIs are 95% percentile intervals from refits of resampled matches, warm-started
  from the full-sample estimate (500 resamples in `run_all.py`, 2,000 in the workstream
  scripts; the first 500 of the 2,000 reproduce `run_all.py`'s intervals to 2e-16). Pooled fits
  resample within season. Held-out metrics use 2,000 paired resamples without refitting, except
  the 2024-25 parameter-uncertainty interval, which refits on 200 resampled training histories.
- Randomness goes through `np.random.default_rng(seed)`; the base seed is 115, each season's
  bootstrap uses 115 + the season's index, the pooled fits use 215 / 216, and every script
  records its seeds in its output's `meta`.
- Outcome probabilities from the 0..10 score grid are renormalised to sum to 1
  (`outcome_probs(normalise=True)`, the default; max missing mass before renormalising 0.6%).
- The single-split forecast in `run_all.py` is a **preliminary stand-in** kept for the
  before/after comparison: 2023-24 strengths applied unchanged to 2024-25, promoted teams given
  the arithmetic mean α and β of the three relegated teams, base rate from seasons before
  2024-25 only. The rolling-origin evaluation (`scripts/run_rolling.py`) is the final one.
