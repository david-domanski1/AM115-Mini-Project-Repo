#!/usr/bin/env python3
"""End-to-end pipeline: download data, fit, bootstrap, simulate, forecast, plot, summarise.

Run with ``python run_all.py`` (or ``make all``).  Writes results/numbers.json,
results/*.csv, results/SUMMARY.md and figures/*.png|pdf.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import figures as F  # noqa: E402
from src.bootstrap import bootstrap_fit, bootstrap_metrics  # noqa: E402
from src.data import ALL_SEASONS, BOOKMAKER_LABELS, CORE_SEASONS, build_matches, season_expected_complete  # noqa: E402
from src.evaluate import base_rate_probs, brier, log_loss, outcome_to_index  # noqa: E402
from src.model import (Spec, fit_poisson, home_advantage_goals, make_design, outcome_probs, rates,  # noqa: E402
                       score_grid, skellam_outcome_probs)
from src.simulate import recovery_study  # noqa: E402

SEED = 115
N_BOOT = 500          # bootstrap replicates per season fit
N_BOOT_POOLED = 500   # bootstrap replicates for the shared-gamma fit
N_BOOT_EVAL = 2000    # paired bootstrap of held-out metrics (no refitting)
N_SIMS = 200          # synthetic seasons per recovery study
COVID = "2020-21"
REF_SEASON = "2023-24"
HOLDOUT_SEASON = "2024-25"
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def fmt_ci(ci, nd=3):
    return f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def main():
    t0 = time.time()
    np.random.seed(SEED)
    F.setup_style()
    numbers = {"meta": {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": SEED, "n_boot_per_season": N_BOOT, "n_boot_pooled": N_BOOT_POOLED,
        "n_boot_eval": N_BOOT_EVAL, "n_sims": N_SIMS,
        "core_seasons": CORE_SEASONS, "extended_seasons_downloaded": ALL_SEASONS,
        "model": "X~Pois(alpha_h*beta_a*gamma), Y~Pois(alpha_a*beta_h); MLE on log scale, L-BFGS-B, analytic gradient",
        "identifiability": "geometric mean of alpha fixed to 1 (one constraint, 40 free parameters per season); "
                           "see diagnostics.two_constraint_fit for the 39-parameter variant",
        "log_loss_units": "nats per match", "brier": "multi-class sum over H/D/A, range 0..2",
        "score_grid": "0..10 goals each side, renormalised to sum to 1 (outcome_probs(normalise=True))",
        "implied_probabilities": "proportional margin removal",
        "bootstrap_seeds": f"per-season fit bootstrap uses seed {SEED} + index of the season in core_seasons; "
                           f"pooled fits use {SEED} + 100 and {SEED} + 101; held-out paired bootstrap uses {SEED}",
        "odds_labels": "football-data.co.uk's non-closing odds are pre-match odds collected on Friday (weekend games) "
                       "or Tuesday (midweek games) afternoons, 1-3 days before kick-off; they are not opening odds",
    }}
    notes = []

    # ------------------------------------------------------------------ data
    print("=" * 78 + "\n[1/6] Data\n" + "=" * 78)
    df = build_matches(ALL_SEASONS)
    core = df[df["season"].isin(CORE_SEASONS)].copy()
    numbers["meta"]["season_info"] = {s: {"n_matches": int(n), "complete": season_expected_complete(s)}
                                      for s, n in core.groupby("season").size().items()}
    partial = [s for s in CORE_SEASONS if not season_expected_complete(s)]
    if partial:
        print(f"[data] partial seasons: {partial}")
    else:
        notes.append("All seasons 2015-16 .. 2025-26 are complete (380 matches each) as of the run date; "
                     "2025-26 was specified as 'current, partial' but the season has finished.")

    # ------------------------------------------------------------ baselines
    print("=" * 78 + "\n[2/6] Baselines\n" + "=" * 78)
    freq = core.groupby("season")["outcome"].value_counts(normalize=True).unstack().reindex(CORE_SEASONS)
    freq["n"] = core.groupby("season").size().reindex(CORE_SEASONS)
    pooled_freq = core["outcome"].value_counts(normalize=True)
    pooled = {k: float(pooled_freq[k]) for k in "HDA"}
    pooled["n"] = int(len(core))
    freq.to_csv(os.path.join(RESULTS, "outcome_frequencies.csv"))
    overround = {}
    for s, g in core.groupby("season"):
        overround[s] = {}
        for key in ["B365", "PS", "PSC", "Avg"]:
            v = g[f"overround_{key}"].dropna()
            overround[s][key] = {"mean": float(v.mean()) if len(v) else None, "n": int(len(v))}
    pd.DataFrame({s: {k: v["mean"] for k, v in d.items()} for s, d in overround.items()}).T.to_csv(
        os.path.join(RESULTS, "overround_by_season.csv"))
    base_p = np.array([pooled[k] for k in "HDA"])
    logloss_base = {}
    for s, g in core.groupby("season"):
        oi = outcome_to_index(g["outcome"])
        row = {"base_rate": log_loss(base_rate_probs(base_p, len(g)), oi), "n": int(len(g))}
        for key in ["PS", "PSC", "B365", "Avg"]:
            p = g[[f"p_{key}_H", f"p_{key}_D", f"p_{key}_A"]].to_numpy()
            ok = ~np.isnan(p).any(axis=1)
            row[key] = log_loss(p[ok], oi[ok]) if ok.sum() else None
            row[f"n_{key}"] = int(ok.sum())
        logloss_base[s] = row
    ll_tab = pd.DataFrame(logloss_base).T
    ll_tab.to_csv(os.path.join(RESULTS, "logloss_baselines_by_season.csv"))
    print(freq.round(3).to_string())
    print("pooled:", {k: round(v, 4) for k, v in pooled.items()})
    print(ll_tab[["base_rate", "PS", "PSC", "n_PS"]].round(4).to_string())
    ps_missing = {s: r["n"] - r["n_PS"] for s, r in logloss_base.items() if r["n_PS"] < r["n"]}
    if ps_missing:
        notes.append(f"Pinnacle odds are missing for some matches: {ps_missing} (season: matches without PSH). "
                     "Pinnacle log-loss and overround for those seasons use only the matches with odds.")
    numbers["baselines"] = {
        "outcome_frequencies": {s: {k: float(freq.loc[s, k]) for k in "HDA"} | {"n": int(freq.loc[s, "n"])}
                                for s in CORE_SEASONS},
        "pooled_frequencies": pooled,
        "overround": overround,
        "overround_labels": BOOKMAKER_LABELS,
        "logloss": logloss_base,
        "logloss_mean_over_seasons": {
            "base_rate": float(ll_tab["base_rate"].mean()),
            "pinnacle_prematch": float(ll_tab["PS"].astype(float).mean()),
            "pinnacle_closing": float(ll_tab["PSC"].astype(float).mean()),
            "note": "base rate here is the pooled 11-season frequency, so it is in-sample for this table; "
                    "the held-out comparison below uses the pre-2024-25 frequency",
        },
    }
    F.fig1_baselines(freq, pooled)

    # ---------------------------------------------------------- Poisson fits
    print("=" * 78 + "\n[3/6] Poisson fits per season + bootstrap\n" + "=" * 78)
    fits, designs, boots = {}, {}, {}
    fit_rows = []
    for s in CORE_SEASONS:
        g = core[core["season"] == s]
        d = make_design(g)
        f = fit_poisson(d)
        assert f.converged, f"{s}: optimiser did not converge"
        b = bootstrap_fit(d, f, n_boot=N_BOOT, seed=SEED + CORE_SEASONS.index(s))   # one seed per season
        f2 = fit_poisson(d, constraint="both")
        ha = home_advantage_goals(f, d)
        fits[s], designs[s], boots[s] = f, d, b
        fit_rows.append({"season": s, "n_matches": d.n_matches, "gamma": f.gamma, "gamma_lo": b["gamma_ci"][0],
                         "gamma_hi": b["gamma_ci"][1], "gamma_se": b["gamma_se"], "ha_goals": ha,
                         "ha_goals_lo": b["ha_goals_ci"][0], "ha_goals_hi": b["ha_goals_ci"][1],
                         "ha_goals_se": b["ha_goals_se"], "mean_home_goals": d.x.mean(), "mean_away_goals": d.y.mean(),
                         "loglik": f.loglik, "n_free_params": f.n_params, "n_iter": f.n_iter,
                         "n_boot": N_BOOT, "n_boot_converged": b["n_converged"],
                         "gamma_two_constraint": f2.gamma, "loglik_two_constraint": f2.loglik})
        print(f"  {s}: γ̂ = {f.gamma:.3f} {fmt_ci(b['gamma_ci'])}   extra home goals = {ha:.3f} "
              f"{fmt_ci(b['ha_goals_ci'])}   (2-constraint variant γ = {f2.gamma:.3f})")
    ha_tab = pd.DataFrame(fit_rows).set_index("season")
    ha_tab.to_csv(os.path.join(RESULTS, "home_advantage_by_season.csv"))
    numbers["poisson_fit_by_season"] = {s: jsonable(r.drop(labels=[]).to_dict()) for s, r in ha_tab.iterrows()}

    # pooled: season-specific alpha/beta, one shared gamma
    dp = make_design(core, season="season")
    fp = fit_poisson(dp)
    assert fp.converged
    strata = pd.factorize(core["season"])[0]
    bp = bootstrap_fit(dp, fp, n_boot=N_BOOT_POOLED, seed=SEED + 100, stratify=strata)
    ha_pooled = home_advantage_goals(fp, dp)
    pooled_gamma = {"gamma": fp.gamma, "gamma_ci": bp["gamma_ci"], "gamma_se": bp["gamma_se"],
                    "ha_goals": ha_pooled, "ha_goals_ci": bp["ha_goals_ci"], "n_matches": dp.n_matches,
                    "n_free_params": fp.n_params, "n_boot": N_BOOT_POOLED, "seasons": CORE_SEASONS,
                    "description": "one gamma shared across seasons, season-specific alpha and beta; "
                                   "bootstrap stratified by season"}
    # pooled excluding 2020-21
    core_nc = core[core["season"] != COVID]
    dnc = make_design(core_nc, season="season")
    fnc = fit_poisson(dnc)
    bnc = bootstrap_fit(dnc, fnc, n_boot=N_BOOT_POOLED, seed=SEED + 101, stratify=pd.factorize(core_nc["season"])[0])
    pooled_gamma_excl_covid = {"gamma": fnc.gamma, "gamma_ci": bnc["gamma_ci"], "ha_goals": home_advantage_goals(fnc, dnc),
                               "ha_goals_ci": bnc["ha_goals_ci"], "n_matches": dnc.n_matches}
    numbers["pooled_gamma"] = pooled_gamma
    numbers["pooled_gamma_excluding_2020_21"] = pooled_gamma_excl_covid
    print(f"  pooled shared γ̂ = {fp.gamma:.3f} {fmt_ci(bp['gamma_ci'])}; excluding 2020-21: {fnc.gamma:.3f} {fmt_ci(bnc['gamma_ci'])}")

    others = [s for s in CORE_SEASONS if s != COVID]
    g_cov, ci_cov = fits[COVID].gamma, boots[COVID]["gamma_ci"]
    g_oth = np.array([fits[s].gamma for s in others])
    ha_cov, ha_ci_cov = ha_tab.loc[COVID, "ha_goals"], boots[COVID]["ha_goals_ci"]
    # bootstrap distribution of the mean of the other seasons' gamma-hats (independent seasons)
    mean_oth_samples = np.mean([boots[s]["gamma_samples"] for s in others], axis=0)
    mean_oth_ci = [float(np.percentile(mean_oth_samples, 2.5)), float(np.percentile(mean_oth_samples, 97.5))]
    diff_samples = boots[COVID]["gamma_samples"] - mean_oth_samples
    diff_ci = [float(np.percentile(diff_samples, 2.5)), float(np.percentile(diff_samples, 97.5))]
    z = (g_cov - g_oth.mean()) / np.sqrt(boots[COVID]["gamma_se"] ** 2 + np.mean([boots[s]["gamma_se"] ** 2 for s in others]) / len(others))
    covid = {"season": COVID, "gamma": g_cov, "gamma_ci": ci_cov, "ha_goals": ha_cov, "ha_goals_ci": ha_ci_cov,
             "mean_gamma_other_seasons": float(g_oth.mean()), "mean_gamma_other_seasons_ci": mean_oth_ci,
             "range_gamma_other_seasons": [float(g_oth.min()), float(g_oth.max())],
             "ci_overlap_with_mean_of_others": bool(ci_cov[1] >= mean_oth_ci[0] and mean_oth_ci[1] >= ci_cov[0]),
             "n_other_seasons_whose_ci_overlaps_2020_21": int(sum(boots[s]["gamma_ci"][0] <= ci_cov[1] and ci_cov[0] <= boots[s]["gamma_ci"][1] for s in others)),
             "gamma_difference_2020_21_minus_mean_others": float(g_cov - g_oth.mean()),
             "gamma_difference_ci": diff_ci, "approx_z": float(z),
             "share_bootstrap_with_gamma_below_1": float(np.mean(boots[COVID]["gamma_samples"] < 1))}
    numbers["covid_comparison"] = covid
    print(f"  {COVID}: γ̂ = {g_cov:.3f} {fmt_ci(ci_cov)} vs mean of other seasons {g_oth.mean():.3f} {fmt_ci(mean_oth_ci)}; "
          f"difference {covid['gamma_difference_2020_21_minus_mean_others']:+.3f} {fmt_ci(diff_ci)}")

    goals_tab = ha_tab[["ha_goals", "ha_goals_lo", "ha_goals_hi"]].rename(columns={"ha_goals": "est", "ha_goals_lo": "lo", "ha_goals_hi": "hi"})
    gamma_tab = ha_tab[["gamma", "gamma_lo", "gamma_hi"]].rename(columns={"gamma": "est", "gamma_lo": "lo", "gamma_hi": "hi"})
    F.fig2_home_advantage(goals_tab, {"est": ha_pooled, "lo": bp["ha_goals_ci"][0], "hi": bp["ha_goals_ci"][1]},
                          kind="goals", n_boot=N_BOOT, name="fig2_home_advantage")
    F.fig2_home_advantage(gamma_tab, {"est": fp.gamma, "lo": bp["gamma_ci"][0], "hi": bp["gamma_ci"][1]},
                          kind="gamma", n_boot=N_BOOT, name="fig2b_home_advantage_gamma")

    # 2023-24 goal distribution, scoreline table, team strengths
    d, f = designs[REF_SEASON], fits[REF_SEASON]
    lam, mu = rates(f.log_alpha, f.log_beta, f.log_gamma, d.home, d.away)
    F.fig3_goals_vs_poisson(d.x, d.y, lam, mu, REF_SEASON)
    grid = score_grid(lam, mu, 10)
    rows = []
    for i in range(4):
        for j in range(4):
            obs = int(np.sum((d.x == i) & (d.y == j)))
            exp = float(grid[:, i, j].sum())
            rows.append({"home_goals": i, "away_goals": j, "scoreline": f"{i}-{j}", "observed": obs,
                         "expected": exp, "ratio_obs_over_exp": obs / exp})
    sc = pd.DataFrame(rows)
    sc.to_csv(os.path.join(RESULTS, "scoreline_table.csv"), index=False)
    ratio = {r["scoreline"]: r["ratio_obs_over_exp"] for r in rows}
    obs_var_home, obs_var_away = float(d.x.var(ddof=1)), float(d.y.var(ddof=1))
    numbers["scorelines_2023_24"] = {
        "season": REF_SEASON, "ratio_0_0": ratio["0-0"], "ratio_1_1": ratio["1-1"], "ratio_1_0": ratio["1-0"],
        "ratio_0_1": ratio["0-1"], "observed_0_0": int(sc.loc[sc.scoreline == "0-0", "observed"].iloc[0]),
        "expected_0_0": float(sc.loc[sc.scoreline == "0-0", "expected"].iloc[0]),
        "observed_1_1": int(sc.loc[sc.scoreline == "1-1", "observed"].iloc[0]),
        "expected_1_1": float(sc.loc[sc.scoreline == "1-1", "expected"].iloc[0]),
        "observed_draw_share": float(np.mean(d.x == d.y)),
        "fitted_draw_share": float(outcome_probs(lam, mu)[:, 1].mean()),
        "home_goals_mean": float(d.x.mean()), "home_goals_var": obs_var_home,
        "away_goals_mean": float(d.y.mean()), "away_goals_var": obs_var_away,
        "grid_truncation_max_missing_mass": float((1 - grid.sum(axis=(1, 2))).max()),
    }
    print(f"  {REF_SEASON} scoreline ratios obs/exp: 0-0 {ratio['0-0']:.2f}, 1-1 {ratio['1-1']:.2f}")
    strengths = pd.DataFrame({"team": f.teams, "alpha_attack": f.alpha, "beta_defensive_weakness": f.beta})
    strengths["beta_relative_to_geo_mean"] = strengths["beta_defensive_weakness"] / np.exp(f.log_beta.mean())
    strengths = strengths.sort_values("alpha_attack", ascending=False).reset_index(drop=True)
    strengths.to_csv(os.path.join(RESULTS, f"team_strengths_{REF_SEASON}.csv"), index=False)
    numbers["team_strengths_2023_24"] = {"gamma": f.gamma, "geo_mean_beta": float(np.exp(f.log_beta.mean())),
                                         "loglik": f.loglik, "n_free_params": f.n_params,
                                         "top_attack": strengths["team"].iloc[0], "bottom_attack": strengths["team"].iloc[-1],
                                         "best_defence": strengths.sort_values("beta_defensive_weakness")["team"].iloc[0]}

    # ---------------------------------------------------------- verification
    print("=" * 78 + "\n[4/6] Verification\n" + "=" * 78)
    rng = np.random.default_rng(SEED)
    lam_r, mu_r = rng.uniform(0.2, 4.0, 200), rng.uniform(0.2, 4.0, 200)
    sk = skellam_outcome_probs(lam_r, mu_r)
    err_wide = float(np.abs(outcome_probs(lam_r, mu_r, max_goals=40) - sk).max())
    err_10 = float(np.abs(outcome_probs(lam_r, mu_r, max_goals=10) - sk).max())
    err_10_draw = float(np.abs(outcome_probs(lam_r, mu_r, max_goals=10)[:, 1] - sk[:, 1]).max())
    print(f"  score grid vs Skellam: max |diff| = {err_wide:.2e} (grid 0..40), {err_10:.2e} (grid 0..10, truncation)")
    assert err_wide < 1e-10
    rec_true = recovery_study(d, f.log_alpha, f.log_beta, f.log_gamma, n_sims=N_SIMS, seed=SEED)
    rec_true["truth_label"] = REF_SEASON
    print(f"  recovery (truth = {REF_SEASON} fit, γ = {rec_true['true_gamma']:.3f}): γ̂ bias {rec_true['gamma_bias']:+.4f}, "
          f"RMSE {rec_true['gamma_rmse']:.4f}; log α RMSE {rec_true['log_alpha_rmse']:.3f}; log β RMSE {rec_true['log_beta_rmse']:.3f}")
    eq_gamma = 1.25
    rec_eq = recovery_study(d, np.zeros(d.n_teams), np.zeros(d.n_teams), np.log(eq_gamma), n_sims=N_SIMS, seed=SEED + 1)
    print(f"  recovery (α = β = 1, γ = {eq_gamma}): γ̂ mean {rec_eq['fitted_gamma'].mean():.4f}, bias {rec_eq['gamma_bias']:+.4f}, "
          f"RMSE {rec_eq['gamma_rmse']:.4f}; log α RMSE {rec_eq['log_alpha_rmse']:.3f}")
    F.fig4_parameter_recovery(rec_true)
    keys = ["n_sims", "true_gamma", "gamma_bias", "gamma_rmse", "gamma_sd", "log_alpha_bias", "log_alpha_rmse",
            "log_beta_bias", "log_beta_rmse", "alpha_rmse", "beta_rmse", "n_converged"]
    numbers["verification"] = {
        "skellam_vs_grid_max_abs_diff_grid40": err_wide,
        "skellam_vs_grid_max_abs_diff_grid10": err_10,
        "skellam_vs_grid_draw_max_abs_diff_grid10": err_10_draw,
        "skellam_check_lambda_mu_range": [0.2, 4.0], "skellam_check_n": 200,
        "recovery_truth_2023_24": {k: rec_true[k] for k in keys} | {"fitted_gamma_mean": float(rec_true["fitted_gamma"].mean())},
        "recovery_equal_strengths": {k: rec_eq[k] for k in keys} | {"fitted_gamma_mean": float(rec_eq["fitted_gamma"].mean()),
                                                                     "fitted_gamma_ci_of_mean": [float(rec_eq["fitted_gamma"].mean() - 1.96 * rec_eq["gamma_sd"] / np.sqrt(N_SIMS)),
                                                                                                 float(rec_eq["fitted_gamma"].mean() + 1.96 * rec_eq["gamma_sd"] / np.sqrt(N_SIMS))]},
    }

    # --------------------------------------------------- preliminary forecast
    print("=" * 78 + "\n[5/6] Preliminary out-of-sample forecast\n" + "=" * 78)
    warnings.warn("PRELIMINARY forecast: 2023-24 strengths applied unchanged to 2024-25; promoted teams get the mean "
                  "strength of the relegated teams. A rolling / time-decayed fit replaces this in the final evaluation.",
                  stacklevel=1)
    hold = core[core["season"] == HOLDOUT_SEASON].copy()
    teams_ref = set(f.teams)
    teams_hold = set(hold["home"]) | set(hold["away"])
    promoted = sorted(teams_hold - teams_ref)
    relegated = sorted(teams_ref - teams_hold)
    rel_alpha = np.mean([f.strength(t)[0] for t in relegated])
    rel_beta = np.mean([f.strength(t)[1] for t in relegated])
    la = {t: np.log(f.strength(t)[0]) for t in teams_ref} | {t: np.log(rel_alpha) for t in promoted}
    lb = {t: np.log(f.strength(t)[1]) for t in teams_ref} | {t: np.log(rel_beta) for t in promoted}
    lam_h = np.exp(hold["home"].map(la).to_numpy() + hold["away"].map(lb).to_numpy() + f.log_gamma)
    mu_h = np.exp(hold["away"].map(la).to_numpy() + hold["home"].map(lb).to_numpy())
    p_pois = outcome_probs(lam_h, mu_h)                      # renormalised 0..10 grid (default)
    p_pois_raw = outcome_probs(lam_h, mu_h, normalise=False)  # legacy: raw grid mass < 1
    p_pin = hold[["p_PS_H", "p_PS_D", "p_PS_A"]].to_numpy()
    p_pinc = hold[["p_PSC_H", "p_PSC_D", "p_PSC_A"]].to_numpy()
    ok = ~np.isnan(p_pin).any(axis=1) & ~np.isnan(p_pinc).any(axis=1)
    if not ok.all():
        notes.append(f"{HOLDOUT_SEASON}: {int((~ok).sum())} matches lack Pinnacle odds and are excluded from the preliminary comparison.")
    hold, p_pois, p_pois_raw, p_pin, p_pinc, lam_h, mu_h = (hold[ok], p_pois[ok], p_pois_raw[ok], p_pin[ok],
                                                            p_pinc[ok], lam_h[ok], mu_h[ok])
    oi = outcome_to_index(hold["outcome"])
    n_hold = len(hold)
    # Headline base rate: H/D/A frequencies from the seasons BEFORE the test season only
    # (2015-16 .. 2023-24), so the baseline sees no test-season data.  The pooled
    # 11-season frequency (which includes 2024-25) is kept only under "legacy".
    base_prior = core[core["season"] < HOLDOUT_SEASON]["outcome"].value_counts(normalize=True)
    base_prior_p = [float(base_prior[k]) for k in "HDA"]
    p_base_prior = base_rate_probs(base_prior_p, n_hold)
    p_base_leaky = base_rate_probs(base_p, n_hold)
    p_uniform = base_rate_probs([1 / 3] * 3, n_hold)
    p_2324 = base_rate_probs([np.mean(d.x > d.y), np.mean(d.x == d.y), np.mean(d.x < d.y)], n_hold)
    sets = {"base_rate": p_base_prior, "poisson": p_pois, "pinnacle": p_pin, "pinnacle_closing": p_pinc,
            "base_rate_all_11_seasons_leaky": p_base_leaky, "base_rate_2023_24_only": p_2324, "uniform": p_uniform,
            "poisson_unnormalised_legacy": p_pois_raw}
    ll = {k: log_loss(p, oi) for k, p in sets.items()}
    br = {k: brier(p, oi) for k, p in sets.items()}
    bll = bootstrap_metrics(sets, oi, log_loss, n_boot=N_BOOT_EVAL, seed=SEED)
    bbr = bootstrap_metrics(sets, oi, brier, n_boot=N_BOOT_EVAL, seed=SEED)
    diffs = {}
    for k in ["poisson", "pinnacle", "pinnacle_closing"]:
        s_ = bll[k]["samples"] - bll["base_rate"]["samples"]
        diffs[k] = {"est": ll[k] - ll["base_rate"], "ci": [float(np.percentile(s_, 2.5)), float(np.percentile(s_, 97.5))]}
    s_ = bll["poisson"]["samples"] - bll["pinnacle"]["samples"]
    diffs["poisson_minus_pinnacle"] = {"est": ll["poisson"] - ll["pinnacle"], "ci": [float(np.percentile(s_, 2.5)), float(np.percentile(s_, 97.5))]}
    s_ = bll["poisson_unnormalised_legacy"]["samples"] - bll["base_rate_all_11_seasons_leaky"]["samples"]
    legacy = {
        "explanation": "Numbers as reported before the critique fixes of 13 Oct 2026: the base rate pooled all 11 "
                       "seasons including the test season, and Poisson probabilities were the raw 0..10 grid mass "
                       "(not renormalised).  Kept so the write-up can quote before/after.",
        "base_rate_frequencies_all_11_seasons": {k: float(v) for k, v in zip("HDA", base_p)},
        "log_loss": {"base_rate_leaky": ll["base_rate_all_11_seasons_leaky"],
                     "poisson_unnormalised": ll["poisson_unnormalised_legacy"]},
        "log_loss_ci": {"base_rate_leaky": bll["base_rate_all_11_seasons_leaky"]["ci"],
                        "poisson_unnormalised": bll["poisson_unnormalised_legacy"]["ci"]},
        "brier": {"base_rate_leaky": br["base_rate_all_11_seasons_leaky"],
                  "poisson_unnormalised": br["poisson_unnormalised_legacy"]},
        "poisson_unnormalised_minus_base_rate_leaky": {
            "est": ll["poisson_unnormalised_legacy"] - ll["base_rate_all_11_seasons_leaky"],
            "ci": [float(np.percentile(s_, 2.5)), float(np.percentile(s_, 97.5))]},
        "max_missing_grid_mass": float((1 - p_pois_raw.sum(axis=1)).max()),
    }
    prelim = {
        "WARNING": "PRELIMINARY stand-in evaluation. Strengths fitted on 2023-24 only, applied unchanged to 2024-25; "
                   "the rolling-origin evaluation in results/rolling.json is the final one.",
        "base_rate_definition": "H/D/A frequencies over 2015-16 .. 2023-24 only (no test-season data)",
        "base_rate_frequencies": {k: v for k, v in zip("HDA", base_prior_p)},
        "train_season": REF_SEASON, "test_season": HOLDOUT_SEASON, "n_test_matches": n_hold,
        "promoted_teams": promoted, "relegated_teams": relegated,
        "promoted_team_strength": {"alpha": float(rel_alpha), "beta": float(rel_beta),
                                   "rule": "arithmetic mean of the three relegated teams' alpha and beta"},
        "log_loss": ll, "log_loss_ci": {k: v["ci"] for k, v in bll.items()},
        "brier": br, "brier_ci": {k: v["ci"] for k, v in bbr.items()},
        "log_loss_paired_differences": diffs, "n_boot": N_BOOT_EVAL,
        "poisson_mean_probs": {k: float(v) for k, v in zip("HDA", p_pois.mean(axis=0))},
        "realised_frequencies": {k: float(np.mean(oi == i)) for i, k in enumerate("HDA")},
        "poisson_mean_lambda": float(lam_h.mean()), "poisson_mean_mu": float(mu_h.mean()),
        "realised_mean_home_goals": float(hold["home_goals"].mean()), "realised_mean_away_goals": float(hold["away_goals"].mean()),
    }
    numbers["preliminary"] = prelim
    numbers["legacy"] = legacy
    pd.DataFrame({"forecast": list(ll), "log_loss": list(ll.values()), "log_loss_lo": [bll[k]["ci"][0] for k in ll],
                  "log_loss_hi": [bll[k]["ci"][1] for k in ll], "brier": list(br.values()),
                  "brier_lo": [bbr[k]["ci"][0] for k in ll], "brier_hi": [bbr[k]["ci"][1] for k in ll]}).to_csv(
        os.path.join(RESULTS, "preliminary_forecast_2024_25.csv"), index=False)
    for k in ["base_rate", "poisson", "pinnacle", "pinnacle_closing"]:
        print(f"  {k:18s} log-loss {ll[k]:.4f} {fmt_ci(bll[k]['ci'], 4)}   Brier {br[k]:.4f}")
    F.fig5_logloss_comparison(
        [("Base rate\n(2015-16..2023-24 H/D/A)", ll["base_rate"], *bll["base_rate"]["ci"]),
         ("Poisson\n(2023-24 strengths)", ll["poisson"], *bll["poisson"]["ci"]),
         ("Pinnacle\n(pre-match odds)", ll["pinnacle"], *bll["pinnacle"]["ci"])],
        [("Poisson − base", diffs["poisson"]["est"], *diffs["poisson"]["ci"]),
         ("Pinnacle − base", diffs["pinnacle"]["est"], *diffs["pinnacle"]["ci"]),
         ("Poisson − Pinnacle", diffs["poisson_minus_pinnacle"]["est"], *diffs["poisson_minus_pinnacle"]["ci"])],
        HOLDOUT_SEASON, n_hold, N_BOOT_EVAL)

    # ------------------------------------------------------------ diagnostics
    numbers["diagnostics"] = {
        "two_constraint_fit": {
            "explanation": "Imposing geometric-mean(alpha)=1 AND geometric-mean(beta)=1 removes one more degree of "
                           "freedom than the model has non-identifiable directions (only alpha->c*alpha, beta->beta/c "
                           "leaves the likelihood unchanged). It forces the geometric mean of mu to equal 1 and pushes the "
                           "away-goal level into gamma. Reported here only to document why the 40-parameter fit is used.",
            "gamma_by_season": {s: float(ha_tab.loc[s, "gamma_two_constraint"]) for s in CORE_SEASONS},
            "loglik_penalty_by_season": {s: float(ha_tab.loc[s, "loglik"] - ha_tab.loc[s, "loglik_two_constraint"]) for s in CORE_SEASONS},
        },
        "runtime_seconds": None, "notes": notes,
    }

    # ---------------------------------------------------------------- summary
    print("=" * 78 + "\n[6/6] Summary\n" + "=" * 78)
    numbers["diagnostics"]["runtime_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, "numbers.json"), "w") as fh:
        json.dump(jsonable(numbers), fh, indent=2)

    overlap_word = "overlap" if covid["ci_overlap_with_mean_of_others"] else "do not overlap"
    ps_over = np.nanmean([overround[s]["PS"]["mean"] for s in CORE_SEASONS if overround[s]["PS"]["mean"] is not None])
    psc_over = np.nanmean([overround[s]["PSC"]["mean"] for s in CORE_SEASONS if overround[s]["PSC"]["mean"] is not None])
    b365_over = np.nanmean([overround[s]["B365"]["mean"] for s in CORE_SEASONS])
    lines = [
        "# Summary: home advantage and Poisson forecasting, Premier League 2015-16 to 2025-26",
        "",
        f"Generated {numbers['meta']['generated']} from football-data.co.uk (E0.csv), seed {SEED}, "
        f"{len(core)} matches in {len(CORE_SEASONS)} seasons (380 each). Runtime {numbers['diagnostics']['runtime_seconds']} s.",
        "",
        "## Home advantage (Poisson MLE, gamma = home-advantage multiplier)",
        "",
        f"- Pooled fit (season-specific attack/defence, one shared gamma, {dp.n_matches} matches): "
        f"gamma = {fp.gamma:.3f}, 95% bootstrap CI {fmt_ci(bp['gamma_ci'])} ({N_BOOT_POOLED} stratified resamples). "
        f"In goals: home teams score {ha_pooled:.3f} {fmt_ci(bp['ha_goals_ci'])} more per match than they would without the multiplier.",
        f"- Same fit excluding 2020-21: gamma = {fnc.gamma:.3f} {fmt_ci(bnc['gamma_ci'])}.",
        f"- Per-season gamma ranges from {ha_tab['gamma'].min():.3f} ({ha_tab['gamma'].idxmin()}) to "
        f"{ha_tab['gamma'].max():.3f} ({ha_tab['gamma'].idxmax()}); all seasons except 2020-21 have a 95% CI entirely above 1."
        if all(boots[s]["gamma_ci"][0] > 1 for s in others) else
        f"- Per-season gamma ranges from {ha_tab['gamma'].min():.3f} ({ha_tab['gamma'].idxmin()}) to "
        f"{ha_tab['gamma'].max():.3f} ({ha_tab['gamma'].idxmax()}). Seasons whose 95% CI includes 1: "
        f"{[s for s in CORE_SEASONS if boots[s]['gamma_ci'][0] <= 1]}.",
        f"- 2020-21 (empty stadiums): gamma = {g_cov:.3f}, 95% CI {fmt_ci(ci_cov)} "
        f"({covid['share_bootstrap_with_gamma_below_1']:.0%} of bootstrap replicates below 1). "
        f"Mean of the other {len(others)} seasons: {g_oth.mean():.3f}, CI {fmt_ci(mean_oth_ci)}. "
        f"The two intervals {overlap_word}; the difference is {covid['gamma_difference_2020_21_minus_mean_others']:+.3f} "
        f"with bootstrap CI {fmt_ci(diff_ci)} (approx. z = {z:.1f}). "
        f"{covid['n_other_seasons_whose_ci_overlaps_2020_21']} of the {len(others)} other seasons have a CI overlapping the 2020-21 CI.",
        f"- Extra home goals per match in 2020-21: {ha_cov:.3f} {fmt_ci(ha_ci_cov)} versus "
        f"{ha_tab.loc[others, 'ha_goals'].mean():.3f} on average in the other seasons.",
        "",
        "## Baselines and forecasting",
        "",
        f"- Pooled outcome frequencies 2015-16..2025-26: home {pooled['H']:.1%}, draw {pooled['D']:.1%}, away {pooled['A']:.1%}. "
        f"2020-21: home {freq.loc[COVID, 'H']:.1%}, draw {freq.loc[COVID, 'D']:.1%}, away {freq.loc[COVID, 'A']:.1%} "
        f"(the only season with more away than home wins)." if freq.loc[COVID, "A"] > freq.loc[COVID, "H"] else
        f"- Pooled outcome frequencies 2015-16..2025-26: home {pooled['H']:.1%}, draw {pooled['D']:.1%}, away {pooled['A']:.1%}. "
        f"2020-21: home {freq.loc[COVID, 'H']:.1%}, draw {freq.loc[COVID, 'D']:.1%}, away {freq.loc[COVID, 'A']:.1%}.",
        f"- Bookmaker overround (mean over matches and seasons): Pinnacle pre-match {ps_over:.2%}, Pinnacle closing {psc_over:.2%}, "
        f"Bet365 {b365_over:.2%}. Per-season values are in results/overround_by_season.csv. 'Pre-match' odds are "
        f"football-data.co.uk's Friday/Tuesday-afternoon collection, 1-3 days before kick-off, not opening odds.",
        f"- Log-loss per season (nats per match), mean over seasons: base rate {ll_tab['base_rate'].mean():.4f} (pooled 11-season "
        f"frequency, in-sample for this table), Pinnacle pre-match {ll_tab['PS'].astype(float).mean():.4f}, "
        f"Pinnacle closing {ll_tab['PSC'].astype(float).mean():.4f}.",
        f"- PRELIMINARY held-out forecast of {HOLDOUT_SEASON} ({n_hold} matches) using 2023-24 strengths unchanged "
        f"(promoted {', '.join(promoted)} given the mean strength of relegated {', '.join(relegated)}). "
        f"Base rate = H/D/A frequencies over 2015-16..2023-24 only; Poisson probabilities renormalised on the 0..10 grid:",
        f"    - log-loss: base rate {ll['base_rate']:.4f} {fmt_ci(bll['base_rate']['ci'], 4)}, "
        f"Poisson {ll['poisson']:.4f} {fmt_ci(bll['poisson']['ci'], 4)}, Pinnacle pre-match {ll['pinnacle']:.4f} {fmt_ci(bll['pinnacle']['ci'], 4)} "
        f"(Pinnacle closing {ll['pinnacle_closing']:.4f} {fmt_ci(bll['pinnacle_closing']['ci'], 4)}).",
        f"    - Brier: base rate {br['base_rate']:.4f}, Poisson {br['poisson']:.4f}, Pinnacle pre-match {br['pinnacle']:.4f}.",
        f"    - Paired differences: Poisson − base {diffs['poisson']['est']:+.4f} {fmt_ci(diffs['poisson']['ci'], 4)}; "
        f"Poisson − Pinnacle {diffs['poisson_minus_pinnacle']['est']:+.4f} {fmt_ci(diffs['poisson_minus_pinnacle']['ci'], 4)}.",
        f"    - Before the 13 Oct fixes (base rate pooled over all 11 seasons incl. 2024-25, Poisson not renormalised): "
        f"base rate {legacy['log_loss']['base_rate_leaky']:.4f}, Poisson {legacy['log_loss']['poisson_unnormalised']:.4f}; "
        f"Poisson − base {legacy['poisson_unnormalised_minus_base_rate_leaky']['est']:+.4f} "
        f"{fmt_ci(legacy['poisson_unnormalised_minus_base_rate_leaky']['ci'], 4)}.",
        "    - This is a stand-in: the rolling-origin evaluation (results/rolling.json) is the final one.",
        "",
        "## Model checks",
        "",
        f"- Parameter recovery on {N_SIMS} simulated seasons (truth = 2023-24 fit, gamma = {rec_true['true_gamma']:.3f}): "
        f"gamma bias {rec_true['gamma_bias']:+.4f}, RMSE {rec_true['gamma_rmse']:.4f}; RMSE of log alpha {rec_true['log_alpha_rmse']:.3f}, "
        f"of log beta {rec_true['log_beta_rmse']:.3f} (natural scale: alpha {rec_true['alpha_rmse']:.3f}, beta {rec_true['beta_rmse']:.3f}).",
        f"- With all teams equal (alpha = beta = 1, planted gamma = {eq_gamma}): mean fitted gamma {rec_eq['fitted_gamma'].mean():.4f}, "
        f"bias {rec_eq['gamma_bias']:+.4f}, RMSE {rec_eq['gamma_rmse']:.4f}.",
        f"- Score-grid P(H/D/A) agrees with the Skellam closed form to {err_wide:.1e} on a 0..40 grid; the 0..10 grid used "
        f"in the pipeline is off by at most {err_10:.1e} because of truncation.",
        f"- 2023-24 scoreline check (observed/expected): 0-0 {ratio['0-0']:.2f}, 1-1 {ratio['1-1']:.2f}, 1-0 {ratio['1-0']:.2f}, 0-1 {ratio['0-1']:.2f}; "
        f"observed draw share {numbers['scorelines_2023_24']['observed_draw_share']:.3f} vs fitted {numbers['scorelines_2023_24']['fitted_draw_share']:.3f}. "
        f"Home goals mean {d.x.mean():.2f} / variance {obs_var_home:.2f}; away {d.y.mean():.2f} / {obs_var_away:.2f}.",
        f"- Free parameters per season: {f.n_params} (20 alpha + 20 beta + gamma, minus one constraint). The two-constraint "
        f"variant in the brief (39 parameters) is over-constrained: for 2023-24 it gives gamma = {ha_tab.loc[REF_SEASON, 'gamma_two_constraint']:.3f} "
        f"with a log-likelihood {ha_tab.loc[REF_SEASON, 'loglik'] - ha_tab.loc[REF_SEASON, 'loglik_two_constraint']:.1f} nats worse, because it "
        f"forces the geometric mean of the away rate to 1 and pushes the away-goal level into gamma.",
        "",
        "## Notes and caveats",
        "",
    ] + [f"- {n}" for n in notes] + [
        "- The per-season base-rate log-loss table uses frequencies from all 11 seasons, so it is in-sample for that table. "
        "The held-out comparison uses only seasons before 2024-25; the old leaky version is in numbers.json under 'legacy'.",
        f"- Bootstrap seeds: per-season fits use seed {SEED} + season index; pooled fits {SEED}+100 / {SEED}+101; "
        f"held-out paired bootstrap {SEED}.",
        "",
        "## Where things are",
        "",
        "- results/numbers.json: every number above.  results/*.csv: per-season tables, scorelines, team strengths.",
        "- figures/fig1_baselines: outcome shares by season.  fig2_home_advantage (extra goals) and fig2b (gamma): headline.",
        "- figures/fig3_goals_vs_poisson: 2023-24 goal histograms vs fitted Poisson.  fig4_parameter_recovery.  fig5_logloss_comparison.",
    ]
    summary = "\n".join(lines)
    with open(os.path.join(RESULTS, "SUMMARY.md"), "w") as fh:
        fh.write(summary + "\n")
    print(summary)
    print(f"\nDone in {time.time() - t0:.1f} s. Results in results/, figures in figures/.")


if __name__ == "__main__":
    main()
