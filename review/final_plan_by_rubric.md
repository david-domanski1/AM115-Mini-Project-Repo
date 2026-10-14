# Final-week plan: the three critiques, merged and mapped to the rubric

Alexander, 13 October 2026. Due 16 October.

Inputs: my Assumptions critique (`review/TEAM_SUMMARY.md`, `review/adjudication_assumptions.md`),
the Estimation-and-uncertainty summary (Claude + ChatGPT second opinion), and Alexia's email of
13 October with two Validation checks (Claude, then a second-model review of that critique).
Rubric: the 75-point final. Everything below was re-read against `src/`, `run_all.py`,
`results/numbers.json` and the football-data notes file; points I could not re-check are marked.

## 1. Where the three critiques say the same thing (do once, one owner)

| Item | Raised by | Verified | Owner |
|---|---|---|---|
| Headline base rate uses all 11 seasons incl. the test season (`run_all.py:316`, `p_base`); the clean `base_rate_pre_2024_25_only` (1.0811) is already computed and unused by Fig 5 | all three | yes | David |
| Outcome probabilities not renormalised after the 0..10 grid (`src/model.py:199`, default `normalise=False`); affects Brier as well as log-loss; use `normalise=True` or Skellam | all three | yes, max missing mass 0.6% | David |
| One γ for all seasons: my LR test 22.0 on 10 df (p = 0.015) vs Alexia's Cochran Q 14.3 on 9 df (p = 0.11) on the 10 non-COVID seasons. **Both right, and they reconcile:** heterogeneity is rejected only when 2020-21 is in the set. Report: "no evidence of between-season variation among the 10 crowd seasons (Q p = 0.11); adding 2020-21 rejects a common γ (p ≈ 0.005–0.015)". Relabel the pooled interval as a common-γ interval. | all three | yes (Q numbers from Alexia's and David's streams agree with each other; I did not recompute τ) | Alexander |
| Same bootstrap seed (115) for every season (`src/bootstrap.py:13`); seed per season | mine, estimation stream | yes, effect nil on the CI | Alexander |
| Closed-door matches are a match-level condition: 92 restart matches sit in 2019-20; Dec 2020 and May 2021 had limited crowds | mine, Alexia check 1 | yes (dates in data; attendance not in data, so flag by date and say so) | Alexander (already planned) |
| Forecast freezes 2023-24 strengths for a year and imputes promoted teams as the mean of the relegated three; both need stating and a sensitivity check | mine, estimation stream, Alexia check 2 (B, C) | yes (`run_all.py`, forecast section) | David |
| Held-out CIs resample only the 380 test matches and treat the fitted parameters as fixed (`src/bootstrap.py:54-68`) | estimation stream, Alexia check 2 (A) | yes | David |
| 2025-26 Pinnacle odds missing for 170 of 380 matches; restrict that row to the 210 matches with odds and say so | estimation stream, Alexia | yes (`n_PS = 210` in `results/logloss_baselines_by_season.csv`) | David |
| The recovery study checks the optimiser under the model, not the model; report its numbers (bias +0.008 = 1.6 MC SE; recovery SD 0.069 vs bootstrap SE 0.070) | mine, Alexia check 1 | yes (`numbers.json`, `verification.recovery_truth_2023_24`) | Alexander |

## 2. Things only Alexia's checks raised, with my verdict

| Item | Verdict |
|---|---|
| **"Pinnacle opening odds" is mislabelled.** football-data's notes file says the non-closing odds are "pre-closing odds" collected "Friday afternoons for weekend games, Tuesday afternoons for midweek games" (notes.txt line 198). `BOOKMAKER_LABELS` in `src/data.py`, `results/SUMMARY.md` and slide 3 all say "opening". | **Right, new, cheap.** Relabel as "Pinnacle pre-match (1-3 days before kickoff)". Also state that the bookmaker knows injuries and transfers the model frozen in May 2024 does not. David. |
| Fig 5 right panel title says "negative = better than base rate" but the third bar is Poisson − Pinnacle (`src/figures.py:204`). | **Right.** Fix the title. Alexia (figures). |
| The forecast is miscalibrated in the large: mean predicted home goals 1.80 vs realised 1.51; P(draw) 0.204 vs 0.245 realised (`numbers.json`, `preliminary`). Log-loss alone hides it. | **Right.** Add calibration-in-the-large per outcome and a 5-bin reliability table for the rolling fit. David. Note for Alexia: the 2023-24 in-sample draw share (0.203 vs 0.216) motivates Dixon–Coles, but pooled over 11 seasons 0-0 is *over*-predicted (263 vs 283) and ρ ≈ −0.05 gains about 5 nats; expect a tiny change and say so. |
| Scoreline table has no uncertainty; the 0-0 ratio 0.67 is 11 vs 16.4, about 1.3 Poisson SDs, one of 16 cells. | **Right.** Add Pearson residuals (observed − expected)/√expected to `results/scoreline_table.csv` and say no cell exceeds 2 SD if that holds. Alexia. |
| Fig 3 variance-vs-mean is not a dispersion test. | **Right, and already answered:** my Pearson dispersion after the fit is 0.88–1.09 across seasons. Put that number in the write-up instead of Fig 3's caption claim. Alexander. |
| Goodness-of-fit checks use 2023-24 only, the highest-scoring season. | **Right.** My pooled 11-season scoreline table already exists in `review/`; use it. Alexander supplies, Alexia plots. |
| Recovery refits are warm-started at the truth (`src/simulate.py:32-36`). | **Right, minor.** Likelihood is concave; one cold-start run (x0 = default) removes the doubt. Alexander. |
| No CI coverage check: are 95% of bootstrap intervals covering the true γ? | **Right, new, and the most direct validation of every error bar in Fig 2.** 200 simulated seasons × 200 bootstrap refits ≈ 40k warm-started fits, a few minutes. Alexander. |
| No misspecified-DGP recovery (negative binomial, correlated goals, drifting strengths). | Right but **set aside** for time; say so in "what this does not show". |
| Independent-match bootstrap understates the held-out CI; use a block bootstrap. | **Unproven, as check 2 says.** The half-season γs differ by noise (estimation stream) and the iid bootstrap is valid for the finite-season mean. A round-block bootstrap of γ for 2020-21 is cheap (I listed it as open on 8 Oct); do it and report whichever way it comes out. Alexander. |
| 2024-25 (γ = 1.065) is nearly as low as 2020-21; z ≈ −3.0 vs −3.3 under a common γ, −2.3 vs −2.5 allowing τ ≈ 0.056. | **Numbers not reproduced by me** (my rough check agrees within rounding). The point stands regardless: the write-up must address 2024-25 directly, or a reader asks why one low season means "no crowds" and the other "noise". Alexander. |
| Fig 2 (extra goals) and Fig 2b (γ) are one finding shown twice: `ha_goals = mean_home_goals × (1 − 1/γ)`. | **Right, and identical to my goals-ratio identity** (γ̂ = ΣH/ΣA, so ha = mean H − mean A). Keep one figure, state the identity. Alexander. |
| Fixture test passes a list with one duplicated and one missing pair (`tests/test_data.py:49-50`); no raw-file hashes; `season_expected_complete` uses `today()` (`src/data.py:78`). | Right, small. Add a uniqueness assertion (I already planned the double-round-robin assertion). Hashes and run-date: set aside; AM115 is graded on the notebook standard, not the repository standard. |
| No boundary or gradient-norm check on L-BFGS-B fits (`src/model.py:23`, bounds ±5; only `res.success` is kept). | Right, cheap: record max |θ| and the projected gradient per fit; report "no parameter within 1 of a bound". Alexander. |
| 500 bootstrap replicates is coarse for 2.5% tails. | Minor. Use 2000 for the headline intervals if runtime allows (about 3× the current 54 s). Alexander. |

## 3. Mapped to the rubric: what each section needs before Friday

**Question (8).** Rephrase: "how many extra home goals per match, and was 2020-21 lower than the other seasons?" Drop "vanish". State the quantity and that the model produces it (Alexander).

**Assumptions (8).** Rubric wants the chosen ones plus at least two implicit ones with where each breaks. We have more than two, verified: (i) matches independent, so the bootstrap resamples matches, breaks for season-level shocks; (ii) strengths frozen all season, breaks visibly in the 2024-25 forecast (1.80 vs 1.51 goals); (iii) promoted = mean of relegated; (iv) crowd status equals season label, breaks for the 92 restart matches; (v) the grid truncation. Plus the goals-ratio identity: the 40-parameter model adds nothing to the point estimate on slide 2 (Alexander).

**Model (6).** Add the identity γ̂ = ΣH/ΣA for a balanced schedule and define the constraint. Already precise otherwise (Alexander).

**Estimation and uncertainty (10).** Every number with an error bar and how it was obtained. Gaps now: team strengths have no CIs (keep the α, β bootstrap draws); the forecast intervals carry no parameter uncertainty; the pooled interval is a common-γ interval; Cochran's Q reported; recovery numbers reported, not "the fit gets them back"; seed per season (David for the forecast, Alexander for the rest).

**Verification (8).** Rubric wants the code checked against known mathematics. We have the Skellam check (1.6e-15) and the gradient check. Add the analytic solution we now know: γ̂ equals the home/away goals ratio to five decimals in all 11 seasons, and the CI coverage simulation (Alexander). This is the cheapest section to move from half to full.

**Validation and a baseline (12).** The biggest section and the weakest now: one split, a leaked base rate, unnormalised probabilities, mislabelled odds, miscalibrated forecast. Fix: pre-test-season base rate, normalised probabilities, rolling-origin over 2016-17 to 2025-26 with per-season Poisson − Pinnacle differences and the detectable gap (SE ≈ 0.012 per season), calibration table, correct odds label, time-decayed fit as the planned improvement, promoted-team sensitivity (David). Fig 5 title (Alexia).

**What changed after section (4).** Needs its own section listing the M2 feedback and the three critique streams, each taken up or set aside with a reason. Set aside with reasons: misspecified-DGP simulation (time), random-effects model beyond reporting Q (one extra number suffices), raw-file hashes (notebook standard), other-league restart comparison if not finished (Alexander writes it, everyone adds their lines).

**What this does not show (4).** The data cannot decide the crowd cause (92 matches, no attendance variable); 2024-25 is as low with full stadiums; one test season cannot separate model from benchmark (gap +0.0135 [−0.010, +0.037] is "cannot tell apart", not "equal"); what another month buys: coverage under misspecification, other leagues, a within-season update (Alexander drafts, David adds the forecast lines).

**Contribution statement.** Use the owner column above; name files and notebook sections per person.

**Implementation (12, AM115 notebook standard).** Runs top to bottom from a clean kernel, headings in rubric order, functions not pasted cells, every figure from the cell above, randomness through an `rng` argument with the seed recorded. The seed-per-season fix fits here. This is Alexia's notebook; everyone's code must be callable as functions (Alexia).

## 4. Order of work, given three days

1. Wording and labels (hours): slide 2 caption, "vanish", "opening" odds, Fig 5 title, common-γ label, recovery numbers.
2. One-line code fixes (hours): pre-test base rate, `normalise=True`, seed per season, fixture uniqueness, boundary report.
3. Cheap computations (half a day each): CI coverage, cold-start recovery, round-block bootstrap for 2020-21, Pearson residuals on the scoreline table, calibration table.
4. David's rolling-origin evaluation with parameter uncertainty and promoted-team sensitivity: the main remaining work, and the 12-point section.
5. Dixon–Coles (Alexia): fit ρ free, expect ≈ −0.05 and a tiny gain, report it honestly.
6. Set aside and say so: misspecified DGP, hashes, random-effects model.
