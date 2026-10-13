# Summary: home advantage and Poisson forecasting, Premier League 2015-16 to 2025-26

Generated 2026-10-13T15:53:30 from football-data.co.uk (E0.csv), seed 115, 4180 matches in 11 seasons (380 each). Runtime 24.4 s.

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

## Where things are

- results/numbers.json: every number above.  results/*.csv: per-season tables, scorelines, team strengths.
- figures/fig1_baselines: outcome shares by season.  fig2_home_advantage (extra goals) and fig2b (gamma): headline.
- figures/fig3_goals_vs_poisson: 2023-24 goal histograms vs fitted Poisson.  fig4_parameter_recovery.  fig5_logloss_comparison.
