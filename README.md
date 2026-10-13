# Home advantage and Poisson forecasting of Premier League matches

AM115 (Harvard, mathematical modelling) mini-project. Everything is reproducible from a
fresh clone with an internet connection:

```bash
pip install -r requirements.txt
make all          # = python run_all.py  followed by  python -m pytest -q tests
```

The pipeline takes about one minute on a laptop (most of it is 5,500 bootstrap refits and
400 simulated-season refits). Raw CSVs are cached in `data/raw/` and never re-downloaded.
Python 3, numpy / pandas / scipy / matplotlib only.

## The model

For a match between home team *h* and away team *a*:

    X ~ Poisson(λ),  Y ~ Poisson(μ),  independent
    λ = α_h · β_a · γ        μ = α_a · β_h

α = attack strength, β = defensive weakness, γ = common home-advantage multiplier.
Parameters are fitted by maximum likelihood on the log scale with `scipy.optimize.minimize`
(L-BFGS-B) and an analytic gradient. Outcome probabilities P(H), P(D), P(A) come from summing
the independent-Poisson score grid over 0..10 goals each side; the Skellam closed form for the
goal difference is implemented as a cross-check.

**Identifiability.** The likelihood is unchanged by α → cα, β → β/c and by nothing else, so the
model has exactly one non-identifiable direction. The fit therefore imposes a single
constraint, geometric mean of α = 1, leaving **40 free parameters per season** (20 α + 20 β +
γ − 1). The brief also suggested fixing the geometric mean of β to 1; that second constraint
removes a degree of freedom the model actually has (it forces the geometric mean of the away
rate μ to be exactly 1) and pushes the league's away-scoring level into γ. For 2023-24 the
two-constraint fit gives γ = 1.64 instead of 1.22 with a log-likelihood 24 nats worse. The
two-constraint fit is still computed for every season and stored under
`diagnostics.two_constraint_fit` in `results/numbers.json` so the difference is documented.

## Data

Source: [football-data.co.uk](https://www.football-data.co.uk), file
`mmz4281/{SS}/E0.csv` for seasons 2005-06 through 2025-26. Columns used: `Date, HomeTeam,
AwayTeam, FTHG, FTAG, FTR`, Bet365 odds `B365H/D/A`, Pinnacle pre-match `PSH/D/A` (from 2012-13),
Pinnacle closing `PSCH/D/A`, and the market average (`BbAvH/D/A` before 2019-20, `AvgH/D/A`
after). The non-closing odds are **not opening odds**: football-data.co.uk's notes say they are
collected on Friday afternoons (weekend games) and Tuesday afternoons (midweek games), 1-3 days
before kick-off, so the bookmaker already knows injuries and selection news that a model fitted
earlier does not. Dates in both `dd/mm/yy` and `dd/mm/yyyy` are parsed. `data/team_names.csv` is the
raw → canonical team-name table (no club changes spelling across these files, so it is an
identity map with full names attached). Implied probabilities use proportional margin
removal, p_k = (1/o_k) / Σ_j (1/o_j); the overround Σ_j 1/o_j − 1 is stored per match.

The core analysis uses 2015-16 to 2025-26 (11 seasons, 4,180 matches). As of the run date
2025-26 has finished, so all 11 seasons are complete; the code still handles a partial season
(it prints the match count instead of asserting 380). Pinnacle odds are absent for 170 of the
380 matches of 2025-26 in the source file; Pinnacle statistics for that season use the 210
matches that have them.

## Layout

| Path | Contents |
|---|---|
| `run_all.py` / `Makefile` | end-to-end driver |
| `src/data.py` | download, cache, clean, team-name table, implied probabilities |
| `src/model.py` | log-likelihood, gradient, constrained parametrisation, fit, score grid, Skellam |
| `src/bootstrap.py` | weighted-refit bootstrap (warm-started) and paired metric bootstrap |
| `src/simulate.py` | synthetic seasons and the parameter-recovery study |
| `src/evaluate.py` | log-loss (nats per match) and multi-class Brier score |
| `src/figures.py` | one function per figure; PNG at 300 dpi and PDF |
| `tests/` | pytest: Skellam vs grid, gradient, recovery, 380-row asserts, implied probabilities |
| `results/numbers.json` | every reported number |
| `results/SUMMARY.md` | plain-English summary (also printed at the end of the run) |
| `notebooks/exploration.ipynb` | thin notebook that calls `src/` |

## Where each figure and number comes from

| Output | Produced by | Contents |
|---|---|---|
| `figures/fig1_baselines` | `figures.fig1_baselines` | home / draw / away share by season, 2020-21 marked |
| `figures/fig2_home_advantage` | `figures.fig2_home_advantage(kind="goals")` | **headline**: extra home goals per match λ − λ/γ with 95% bootstrap CI, pooled shared-γ band |
| `figures/fig2b_home_advantage_gamma` | same, `kind="gamma"` | γ̂ by season with CIs, line at γ = 1 |
| `figures/fig3_goals_vs_poisson` | `figures.fig3_goals_vs_poisson` | 2023-24 goals per team per match vs fitted Poisson mixture |
| `figures/fig4_parameter_recovery` | `figures.fig4_parameter_recovery` | fitted vs true log α, log β over 200 simulated seasons; histogram of γ̂ |
| `figures/fig5_logloss_comparison` | `figures.fig5_logloss_comparison` | preliminary held-out 2024-25 log-loss: base rate, Poisson, Pinnacle, with paired-bootstrap CIs |
| `results/outcome_frequencies.csv` | step 2 | per-season H/D/A shares |
| `results/overround_by_season.csv` | step 2 | mean overround per bookmaker per season |
| `results/logloss_baselines_by_season.csv` | step 2 | base-rate and bookmaker log-loss per season |
| `results/home_advantage_by_season.csv` | step 3 | γ̂, CIs, extra goals, log-likelihood, two-constraint diagnostic |
| `results/scoreline_table.csv` | step 3 | observed vs expected counts for scorelines 0-0 .. 3-3 in 2023-24 |
| `results/team_strengths_2023-24.csv` | step 3 | α̂, β̂ per team, sorted by α̂ |
| `results/preliminary_forecast_2024_25.csv` | step 5 | log-loss and Brier with CIs for every forecaster |

`results/numbers.json` keys: `baselines`, `poisson_fit_by_season`, `pooled_gamma`,
`pooled_gamma_excluding_2020_21`, `covid_comparison`, `scorelines_2023_24`,
`team_strengths_2023_24`, `verification`, `preliminary`, `diagnostics`, `meta`.

## Conventions

- Seasons are labelled `2015-16` everywhere. Log-loss is always in nats per match. The Brier
  score is the multi-class sum over H/D/A (range 0 to 2).
- Bootstrap CIs are 95% percentile intervals from 500 resamples of the 380 matches of a
  season (refit each time, warm-started from the full-sample estimate); the pooled fit is
  bootstrapped with resampling stratified by season. Held-out metrics use 2,000 paired
  resamples without refitting.
- Randomness goes through `np.random.default_rng(seed)`; the base seed is 115, each season's
  bootstrap uses 115 + the season's index, the pooled fits use 215 / 216, and every script
  records its seed in its output.
- Outcome probabilities from the 0..10 score grid are renormalised to sum to 1
  (`outcome_probs(normalise=True)`, the default).
- The single-split forecast in `run_all.py` is a **preliminary stand-in**: 2023-24 strengths
  applied unchanged to 2024-25, promoted teams given the arithmetic mean α and β of the three
  relegated teams; its base rate uses only seasons before 2024-25. The rolling-origin
  evaluation (`scripts/run_rolling.py`) is the final one.
