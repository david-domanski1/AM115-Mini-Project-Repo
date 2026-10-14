# Summary: home advantage and Poisson forecasting, Premier League 2015-16 to 2025-26

Generated 2026-10-13T16:39:27 from football-data.co.uk (E0.csv), seed 115, 4180 matches in 11 seasons (380 each). Runtime 26.3 s.

## Home advantage (Poisson MLE, gamma = home-advantage multiplier)

- Pooled fit (season-specific attack/defence, one shared gamma, 4180 matches): gamma = 1.217, 95% bootstrap CI [1.177, 1.261] (500 stratified resamples). In goals: home teams score 0.276 [0.228, 0.326] more per match than they would without the multiplier.
- Same fit excluding 2020-21: gamma = 1.239 [1.192, 1.285].
- Per-season gamma ranges from 1.008 (2020-21) to 1.341 (2022-23). Seasons whose 95% CI includes 1: ['2020-21', '2024-25'].
- 2020-21 (empty stadiums): gamma = 1.008, 95% CI [0.877, 1.134] (45% of bootstrap replicates below 1). Mean of the other 10 seasons: 1.244, CI [1.198, 1.293]. The two intervals do not overlap; the difference is -0.236 with bootstrap CI [-0.374, -0.093] (approx. z = -3.3). 7 of the 10 other seasons have a CI overlapping the 2020-21 CI.
- Extra home goals per match in 2020-21: 0.011 [-0.178, 0.172] versus 0.303 on average in the other seasons.

## Baselines and forecasting

- Pooled outcome frequencies 2015-16..2025-26: home 44.3%, draw 23.7%, away 32.0%. 2020-21: home 37.9%, draw 21.8%, away 40.3% (the only season with more away than home wins).
- Bookmaker overround (mean over matches and seasons): Pinnacle pre-match 2.71%, Pinnacle closing 2.51%, Bet365 4.51%. Per-season values are in results/overround_by_season.csv. 'Pre-match' odds are football-data.co.uk's Friday/Tuesday-afternoon collection, 1-3 days before kick-off, not opening odds.
- Log-loss per season (nats per match), mean over seasons: base rate 1.0664 (pooled 11-season frequency, in-sample for this table), Pinnacle pre-match 0.9569, Pinnacle closing 0.9538.
- PRELIMINARY held-out forecast of 2024-25 (380 matches) using 2023-24 strengths unchanged (promoted Ipswich, Leicester, Southampton given the mean strength of relegated Burnley, Luton, Sheffield United). Base rate = H/D/A frequencies over 2015-16..2023-24 only; Poisson probabilities renormalised on the 0..10 grid:
    - log-loss: base rate 1.0811 [1.0539, 1.1070], Poisson 0.9837 [0.9292, 1.0411], Pinnacle pre-match 0.9703 [0.9213, 1.0190] (Pinnacle closing 0.9664 [0.9172, 1.0159]).
    - Brier: base rate 0.6557, Poisson 0.5878, Pinnacle pre-match 0.5788.
    - Paired differences: Poisson − base -0.0974 [-0.1405, -0.0522]; Poisson − Pinnacle +0.0134 [-0.0097, 0.0370].
    - Before the 13 Oct fixes (base rate pooled over all 11 seasons incl. 2024-25, Poisson not renormalised): base rate 1.0803, Poisson 0.9838; Poisson − base -0.0965 [-0.1398, -0.0513].
    - This is a stand-in: the rolling-origin evaluation (results/rolling.json) is the final one.

## Model checks

- Parameter recovery on 200 simulated seasons (truth = 2023-24 fit, gamma = 1.217): gamma bias +0.0079, RMSE 0.0689; RMSE of log alpha 0.129, of log beta 0.141 (natural scale: alpha 0.130, beta 0.188).
- With all teams equal (alpha = beta = 1, planted gamma = 1.25): mean fitted gamma 1.2533, bias +0.0033, RMSE 0.0793.
- Score-grid P(H/D/A) agrees with the Skellam closed form to 8.9e-16 on a 0..40 grid; the 0..10 grid used in the pipeline is off by at most 8.9e-04 because of truncation.
- 2023-24 scoreline check (observed/expected): 0-0 0.67, 1-1 1.10, 1-0 0.83, 0-1 0.87; observed draw share 0.216 vs fitted 0.203. Home goals mean 1.80 / variance 1.86; away 1.48 / 1.63.
- Free parameters per season: 40 (20 alpha + 20 beta + gamma, minus one constraint). The two-constraint variant in the brief (39 parameters) is over-constrained: for 2023-24 it gives gamma = 1.636 with a log-likelihood 24.1 nats worse, because it forces the geometric mean of the away rate to 1 and pushes the away-goal level into gamma.

## Notes and caveats

- All seasons 2015-16 .. 2025-26 are complete (380 matches each) as of the run date; 2025-26 was specified as 'current, partial' but the season has finished.
- Pinnacle odds are missing for some matches: {'2025-26': 170} (season: matches without PSH). Pinnacle log-loss and overround for those seasons use only the matches with odds.
- The per-season base-rate log-loss table uses frequencies from all 11 seasons, so it is in-sample for that table. The held-out comparison uses only seasons before 2024-25; the old leaky version is in numbers.json under 'legacy'.
- Bootstrap seeds: per-season fits use seed 115 + season index; pooled fits 115+100 / 115+101; held-out paired bootstrap 115.

## Workstream results (each number: estimate, 95% interval, how obtained)

### Home advantage over time, 2005-06..2025-26 (scripts/run_timeseries.py)

Generated 2026-10-13T16:20:52, seed 115, runtime 432 s. Full table of 49 headline entries in results/timeseries.json and review/results_timeseries.md.

- common gamma, 20 crowd seasons (excl. 2020-21): 1.301 [1.270, 1.337] (n = 7600; pooled MLE, season-specific strengths and one shared gamma over the 20 seasons 2005-06..2025-26 except 2020-21 (= sum H / sum A); common-gamma interval: percentile CI from 2000 season-stratified bootstrap refits)
- extra home goals per match, same fit: 0.359 [0.326, 0.396] (n = 7600; pooled common-gamma fit, mean(lambda - lambda/gamma); season-stratified bootstrap percentile CI)
- gamma 2020-21 (closed doors): 1.008 [0.884, 1.143] (n = 380; per-season MLE (= sum H / sum A); percentile CI from 2000 iid match-bootstrap refits, seed 120)
- gamma 2024-25 (full stadiums): 1.065 [0.956, 1.187] (n = 380; per-season MLE (= sum H / sum A); percentile CI from 2000 iid match-bootstrap refits, seed 124)
- verification: max |gamma-hat - sum H / sum A|: 0.000000353 [0.000000007, 0.000000353] (n = 21; max over 21 seasons of |gamma_hat (L-BFGS-B) - sum home goals / sum away goals|; 'ci' is [min, max] over seasons, not a confidence interval; pooled fits: max 3.4e-07; extra-goals identity: max 3.8e-07)
- Cochran Q, 10 core crowd seasons: 12.77 [2.70, 19.02] (n = 9; Cochran's Q on log gamma_hat with bootstrap SEs (2000 reps), df = n = 9, p = 0.1732; 'ci' is the central 95% range of chi2_9 under a common gamma)
- Cochran Q, 11 core seasons (adds 2020-21): 21.78 [3.25, 20.48] (n = 10; Cochran's Q on log gamma_hat with bootstrap SEs (2000 reps), df = n = 10, p = 0.0163; 'ci' is the central 95% range of chi2_10 under a common gamma)
- Cochran Q, 20 crowd seasons since 2005-06: 37.74 [8.91, 32.85] (n = 19; Cochran's Q on log gamma_hat with bootstrap SEs (2000 reps), df = n = 19, p = 0.0064; 'ci' is the central 95% range of chi2_19 under a common gamma)
- trend in log gamma per season, 20 crowd seasons: -0.0099 [-0.0142, -0.0055] (n = 20; inverse-variance weighted regression of log gamma_hat on season start year (bootstrap SEs), SE inflated by sqrt(residual Q/df) if > 1; p = 7.9e-06; residual Q = 17.8 on 18 df (p = 0.471))
- z of 2020-21 vs other seasons, random effects: -2.82 [-4.78, -0.86] (n = 20; (log gamma_s - DL random-effects mean of 20 other crowd seasons) / sqrt(se_s^2 + tau^2 + se_mean^2), tau = 0.0603; p = 0.0049; 'ci' = z +- 1.96)
- z of 2024-25 vs other seasons, random effects: -3.03 [-4.99, -1.07] (n = 19; (log gamma_s - DL random-effects mean of 19 other crowd seasons) / sqrt(se_s^2 + tau^2 + se_mean^2), tau = 0.0373; p = 0.0024; 'ci' = z +- 1.96)
- gamma closed doors - full crowds, PL 2019-20..2021-22 (date flag): -0.108 [-0.247, 0.040] (n = 442; difference of the two gammas in the same crowd-status fit; percentile CI of the paired bootstrap differences (2000 season-stratified refits))
- coverage of the 95% bootstrap percentile interval: 0.948 [0.925, 0.966] (n = 500; share of 500 simulated 2023-24-style seasons (truth gamma = 1.2171) whose 95% bootstrap percentile interval (200 refits) contains the truth; Clopper-Pearson 95% CI)
- max |theta| over all fits (bounds +-5): 3.24 [0.00, 3.24] (n = 170937; largest |log-parameter| over every fit in this script (n = number of fits); bound is +-5; 'ci' is [0, max]; no parameter within 1.76 of a bound)

### Rolling-origin forecast evaluation, test seasons 2016-17..2025-26 (scripts/run_rolling.py)

Generated 2026-10-13T16:19:43, seed 115, runtime 628 s. Full table of 35 headline entries in results/rolling.json and review/results_rolling.md.

- Poisson log-loss, mean of 10 seasons: 0.9662 [0.9506, 0.9822] (n = 3800; rolling Poisson (hl=365d|gamma=season|promoted=relegated), refitted before every Fri-Mon / Tue-Thu window; mean over the 10 test seasons of the per-season mean log-loss (nats/match); 95% CI from 2000 match-resampling bootstrap replicates per season, averaged across seasons replicate by replicate)
- base rate (prior seasons only): 1.0663 [1.0574, 1.0749] (n = 3800; H/D/A frequencies of all seasons before the test season (from 2005-06); mean over the 10 test seasons of the per-season mean log-loss (nats/match); 95% CI from 2000 match-resampling bootstrap replicates per season, averaged across seasons replicate by replicate)
- Pinnacle pre-match (odds matches): 0.9491 [0.9333, 0.9660] (n = 3630; Pinnacle pre-match (Fri/Tue collection, 1-3 days before kick-off), margin removed proportionally; Pinnacle and the paired Poisson values use the 3,630 matches with odds (2025-26: 210 of 380); mean over the 10 test seasons of the per-season mean log-loss (nats/match); 95% CI from 2000 match-resampling bootstrap replicates per season, averaged across seasons replicate by replicate)
- Pinnacle closing (odds matches): 0.9459 [0.9302, 0.9629] (n = 3630; Pinnacle closing odds, margin removed proportionally; Pinnacle and the paired Poisson values use the 3,630 matches with odds (2025-26: 210 of 380); mean over the 10 test seasons of the per-season mean log-loss (nats/match); 95% CI from 2000 match-resampling bootstrap replicates per season, averaged across seasons replicate by replicate)
- Poisson - base rate: -0.1001 [-0.1140, -0.0867] (n = 3800; paired difference (negative = Poisson better), all 3,800 matches; mean over the 10 test seasons of the per-season mean log-loss (nats/match); 95% CI from 2000 match-resampling bootstrap replicates per season, averaged across seasons replicate by replicate)
- Poisson - Pinnacle pre-match: 0.0141 [0.0083, 0.0195] (n = 3630; paired difference (positive = Pinnacle better); Pinnacle and the paired Poisson values use the 3,630 matches with odds (2025-26: 210 of 380); mean over the 10 test seasons of the per-season mean log-loss (nats/match); 95% CI from 2000 match-resampling bootstrap replicates per season, averaged across seasons replicate by replicate)
- detectable gap in one season: 0.0249 [0.0208, 0.0322] (n = 10; 2.80 x per-season paired bootstrap SE (5% two-sided test, 80% power); est = mean over seasons, 'ci' = range over the 10 seasons (a design quantity, not a confidence interval); mean SE = 0.0089)
- gain from time decay (no decay - 365-day half-life): 0.0159 [0.0097, 0.0221] (n = 3800; paired log-loss difference no decay - best half-life (positive = decay helps), all 3,800 test matches; 2000 paired match-bootstrap replicates)
- 2024-25 mean lambda - realised home goals, rolling: 0.094 [-0.027, 0.216] (n = 380; rolling mean lambda 1.607 vs realised 1.513; match bootstrap)
- same, frozen 2023-24 fit: 0.287 [0.158, 0.416] (n = 380; frozen 2023-24 fit from results/numbers.json 'preliminary': mean lambda 1.800 vs realised 1.513; CI from 2000 resamples of the 380 matches' home goals (lambda fixed))
- 2024-25 log-loss, parameter + imputation + match uncertainty: 0.9763 [0.9402, 1.0435] (n = 380; 200 replicates: resample every season up to 2024-25 with replacement (within season), redo all 45 rolling refits incl. the promoted-team targets, score the resampled test matches; percentile CI. Includes parameter, imputation and match sampling uncertainty; width ratio to match-only = 1.01)
- flat-stake return at Pinnacle closing, margin 0: 0.068 [-0.053, 0.192] (n = 689; flat 1-unit bets at Pinnacle closing odds when p_model * odds - 1 > 0.00, 2024-25 and 2025-26 (590 matches with odds); return per unit staked; 2000 match-resampling bootstrap replicates)

### Dixon-Coles correction (scripts/run_dixon_coles.py)

Generated 2026-10-13T16:23:48, seed 115, runtime 60 s. Full table of 15 headline entries in results/dixon_coles.json and review/results_dixon_coles.md.

- pooled rho, 11 seasons: -0.0507 [-0.0974, -0.0086] (n = 4180; joint MLE over 11 seasons (season-specific alpha, beta; one gamma; one rho; 431 params), 95% percentile CI from 1000 season-stratified match-resampling refits (seed 415); profile-likelihood CI [-0.0937, -0.0075])
- LR statistic vs rho = 0: 5.27 [0.00, 5.02] (n = 4180; 2 x (loglik DC - loglik Poisson) for the pooled fit, 1 df; p = 0.0216; ci = central 95% range of chi2_1 under rho = 0 (not a CI of the statistic))
- in-sample log-loss change per match, DC - Poisson: -0.00074 [-0.00151, 0.00004] (n = 4180; outcome (H/D/A) log-loss of the pooled DC fit minus the pooled Poisson fit over all 4180 matches, nats per match (negative = DC better); paired bootstrap of matches, 2000 resamples, seed 115, no refit)
- held-out 2024-25 log-loss, DC - Poisson: -0.00078 [-0.00180, 0.00025] (n = 380; paired log-loss difference, DC fitted on 2023-24 minus Poisson, nats per match on 2024-25 (negative = DC better); 2000 paired resamples, seed 115)
- largest |Pearson residual| in the pooled scoreline table: 3.33 [-1.88, 2.01] (n = 4180; largest |(obs-exp)/sqrt(exp)| of the 17 pooled cells (0..3 x 0..3, 4+) under the pooled Poisson fit, cell 2-2; 4 of 17 cells exceed 2; ci = 95% range of that cell's residual in 2000 data sets simulated from the fit (seed 416, no refit); P(max |r| >= observed | model) = 0.009)
- Pearson dispersion, pooled: 0.948 [0.969, 1.031] (n = 8360; sum of squared Pearson residuals of home and away goals / (2m - p), pooled Poisson fit; ci = 95% range under the Poisson model (chi2_7930/7930); per-season values 0.880..1.090 in dixon_coles_by_season.csv)

## Where things are

- results/numbers.json: every number of the run_all.py pipeline.  results/*.csv: per-season tables, scorelines, team strengths.
- results/timeseries.json, rolling.json, dixon_coles.json: the three workstreams (scripts/run_*.py), each with a
  'headline' dict of {est, ci, n, how}; prose notes in review/results_*.md.
- figures/fig1_baselines: outcome shares by season.  fig2_home_advantage_2005_2026: the home-advantage figure (gamma by
  season with the common-gamma band); fig2_home_advantage: the same 11 seasons in extra goals.
- figures/fig3_goals_envelope: pooled goal histograms vs fitted Poisson with a bootstrap band.  fig4_parameter_recovery.
  fig5_logloss_comparison (single split).  fig6_rolling_logloss, fig7_rolling_reliability (rolling evaluation).
  fig8_crowd_gamma (crowd status, three leagues).  fig9_dixon_coles (rho by season, profile likelihood, residuals).
- notebooks/final_report.ipynb: the report, in rubric order, executed from a clean kernel.
