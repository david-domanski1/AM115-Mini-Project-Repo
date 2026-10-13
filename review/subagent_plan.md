# Plan: finish the three slide-3 workstreams with three Opus 5.5 subagents

Alexander, 13 October 2026. Due Friday 16 October.

## Shape

Three agents run in parallel in the same working tree (data/raw and data/processed are
gitignored, so separate worktrees would start without data). Conflicts are avoided by
file ownership: each agent creates its own module, script, results files and figures, and
touches nothing another agent owns. I do a short step 0 first so all three start from the
same fixed base, then integrate at the end.

Checklist

- [ ] Step 0: shared small fixes, run tests, one commit
- [ ] Launch A (forecast), B (home advantage over time), C (Dixon–Coles + notebook) in parallel
- [ ] Integrate: merge results, final notebook run, write-up sections, tests, commit
- [ ] Friday: submit

## Step 0 (me, before launch, under an hour)

Small fixes every critique asked for, so no agent re-does them and all build on them:

1. `src/model.py`: `outcome_probs(..., normalise=True)` as the default; keep the raw option.
2. `run_all.py`: headline hold-out base rate = pre-2024-25 frequencies (`p_base_prior`);
   seed per season for `bootstrap_fit` (SEED + season index); keep the old numbers in a
   `legacy` key so the write-up can quote the before/after.
3. `src/data.py`: `BOOKMAKER_LABELS` "Pinnacle (opening)" → "Pinnacle (pre-match, Fri/Tue
   collection)"; same in `results/SUMMARY.md` text.
4. `src/figures.py:204`: right-panel title → "Paired differences (negative = first forecast better)".
5. `tests/test_data.py`: assert the 380 (home, away) pairs are distinct in every complete season.
6. `src/data.py`: `URL_PATTERN` takes a league code (default `E0`) so agent B can add leagues
   without a merge conflict later; `raw_path` includes the league code.
7. Run `make all`, confirm tests pass, commit "Apply critique fixes shared by all workstreams".

## Shared rules (go in every brief verbatim)

- Work from the repo root with `python3`; numpy, pandas, scipy, matplotlib, pytest are installed.
  `pip install nbformat nbconvert jupyter` is allowed (agent C needs it).
- Randomness only through an `rng = np.random.default_rng(seed)` argument; record the seed in output.
- Every estimated number carries a 95% interval and a one-line statement of how it was obtained.
- Code goes in `src/<module>.py` as functions the notebook can import; the driver is
  `scripts/<name>.py`; outputs go to `results/<name>.json` and `results/<name>_*.csv`,
  figures to `figures/fig<N>_<name>.png|pdf`; a human-readable `review/results_<name>.md`
  with every headline number, its CI, and the rubric section it serves.
- Do not edit `run_all.py`, `results/numbers.json`, `results/SUMMARY.md`, or another agent's
  files. Do not commit. Add at least one pytest test in `tests/test_<name>.py`.
- Each driver script must finish in under 15 minutes on a laptop; cut replicates, not scope.
- Final report back: list of files written, headline numbers with CIs, anything set aside and why,
  anything that disagrees with `review/final_plan_by_rubric.md`.

## Agent A: forecast (David's slide-3 box; rubric: Validation 12, Estimation 10)

Owns `src/rolling.py`, `scripts/run_rolling.py`, `tests/test_rolling.py`, `results/rolling*`,
`figures/fig6_*`, `figures/fig7_*`, `review/results_rolling.md`.

1. Rolling-origin evaluation, test seasons 2016-17 to 2025-26. Before each matchday in the
   test season, fit on all prior matches with exponential time decay (half-life grid
   {no decay, 2 years, 1 year, 180 days}), all seasons pooled with season-specific γ off
   (one γ) and on (compare). Promoted teams: three rules (mean of relegated, league-average
   strengths, shrink 50% toward league average); report the sensitivity.
2. Score every test match: Poisson (normalised), base rate from prior seasons only, Pinnacle
   pre-match and closing on the matches that have odds (2025-26: 210 of 380, say so). Log-loss
   and Brier per season, paired bootstrap CIs, Poisson − Pinnacle per season with the
   between-season SD and the detectable gap (SE ≈ 0.012 per season).
3. Calibration: calibration-in-the-large per outcome per season, 5-bin reliability table, and
   mean predicted vs realised goals; show whether the rolling fit removes the 1.80 vs 1.51
   problem of the frozen fit.
4. Parameter uncertainty: for 2024-25, bootstrap the training set (200 replicates) and refit,
   to give a held-out log-loss interval that includes parameter and imputation uncertainty;
   compare its width with the match-only bootstrap.
5. Betting test, kept small: flat stakes at Pinnacle closing odds on 2024-25 and 2025-26 when
   the model's probability exceeds the implied probability by a margin grid {0, 5, 10}%;
   return with bootstrap CI. One table, one sentence.
6. Figures: fig6 per-season log-loss (Poisson, base, Pinnacle) with CIs; fig7 reliability.

## Agent B: home advantage over time (Alexander's box; rubric: Assumptions 8, Verification 8, Estimation 10)

Owns `src/timeseries.py`, `src/crowd.py`, `scripts/run_timeseries.py`, `tests/test_timeseries.py`,
`results/timeseries*`, `results/crowd*`, `results/coverage*`, `figures/fig2_*` (replaces),
`figures/fig8_*`, `review/results_timeseries.md`. May add league codes to `src/data.py` via the
parameter from step 0 and add team-name rows to `data/team_names.csv` (append only).

1. Download 2005-06 to 2014-15 (already in `ALL_SEASONS`), fit γ per season with 500-replicate
   bootstrap CIs for 2005-06 to 2025-26; verify γ̂ = Σ home goals / Σ away goals to 1e-6 every
   season and record it as the verification check.
2. Common-γ tests: Cochran's Q and the likelihood-ratio test with and without 2020-21, and the
   DerSimonian–Laird τ; z of 2020-21 and of 2024-25 against the other seasons under common γ
   and under random effects. One table.
3. Crowd flag by date: closed doors 17 June 2020 to end of 2020-21; limited attendance
   matchdays (2 to 15 December 2020 for the tier-2 clubs, and 18 to 23 May 2021) flagged
   separately; say that attendance is not in the data. Fit γ by crowd status with
   season-specific strengths over 2019-20 to 2021-22; report the restart-vs-rest difference
   with its bootstrap CI (expect it to be wide).
4. Block bootstrap for 2020-21 and 2024-25: blocks = 38 consecutive 10-match date blocks, and
   a team-cluster bootstrap; compare interval widths with the iid bootstrap.
5. CI coverage: 200 simulated 2023-24-style seasons × 200 bootstrap refits; share of 95%
   intervals containing the true γ, with a binomial CI. One cold-start recovery run
   (default x0) to compare with the warm-started one. Record max |θ| and projected-gradient
   norm per fit; state whether any parameter is within 1 of the ±5 bound.
6. Stretch, only if 1–5 are done: Bundesliga (D1) and La Liga (SP1) 2019-20 to 2021-22,
   same crowd-flag fit, as parallel comparisons.
7. Figures: new fig2 (one panel, γ or extra goals with the identity stated, 2005-06 to 2025-26,
   common-γ band excluding 2020-21); fig8 crowd-status γ.

## Agent C: Dixon–Coles and the final notebook (Alexia's box; rubric: Implementation 12, Model 6)

Owns `src/dixon_coles.py`, `scripts/run_dixon_coles.py`, `tests/test_dixon_coles.py`,
`results/dixon_coles*`, `figures/fig3_*` (replaces), `figures/fig9_*`,
`notebooks/final_report.ipynb`, `review/results_dixon_coles.md`.

1. Dixon–Coles τ(x, y; λ, μ, ρ) correction for the 0-0, 0-1, 1-0, 1-1 cells; MLE of ρ with
   sign free, per season and pooled with season-specific strengths; bootstrap CI for ρ
   (200 replicates); likelihood-ratio test against ρ = 0. Expect ρ ≈ −0.05 and report what
   comes out.
2. Scoreline table pooled over the 11 seasons, 0..3 × 0..3 plus "4+", observed, expected under
   Poisson and under Dixon–Coles, Pearson residuals; a parametric-bootstrap envelope for the
   goal histograms (replaces fig3's variance-vs-mean caption).
3. Effect on forecasts: in-sample log-loss change over the 11 seasons, and held-out log-loss on
   the frozen 2023-24 → 2024-25 split with and without the correction (same probabilities
   as `run_all.py`, normalised). Say in one line how many nats it moves.
4. Final notebook `notebooks/final_report.ipynb` in rubric order: Question, Assumptions, Model,
   Estimation and uncertainty, Verification, Validation and a baseline, What changed after
   section, What this does not show, Contribution statement. Every cell calls functions from
   `src/`; every figure is produced by the cell above it; `rng` argument with the seed
   printed; runs top to bottom from a clean kernel via
   `jupyter nbconvert --execute`. Leave clearly marked placeholder cells that import
   `results/rolling.json` and `results/timeseries.json` for agents A and B's sections, with
   the headline numbers rendered from those files, so integration is a re-run.
5. Figure fixes: fig5 title (done in step 0; verify), one home-advantage figure not two.

## Integration (me, Wednesday to Thursday)

1. Run all three driver scripts and `make all`; read each `review/results_*.md`.
2. Update `results/SUMMARY.md` and `README.md`; execute the final notebook from a clean kernel.
3. Write the three rubric sections that are prose only: What changed after section (M2 feedback
   and the three critique streams, taken up or set aside with reasons), What this does not show,
   Contribution statement (per person, with file names).
4. Slides: slide 2 caption, "vanish", odds label, new figures.
5. `make test`, commit, share.

## Risks

- football-data rate-limits (WebFetch got a 429 today; curl worked). Agent B downloads with the
  existing cached downloader; if it fails, retry with a pause rather than hammering.
- Three agents, one tree: if two touch `src/data.py` the second edit is lost. Only B may edit it.
- Runtime: A's rolling refit is at most about 4,000 warm-started fits per half-life; B's coverage
  is 40,000 fits. Both are minutes, not hours, at the current 5 ms per fit.
- Agent C's notebook cannot be finished until A and B report; that is why its cells read from
  result files.
