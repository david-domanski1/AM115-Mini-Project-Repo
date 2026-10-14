#!/usr/bin/env python3
"""Dixon–Coles low-score correction: estimate rho, test it, and measure what it changes.

Run from the repo root:  python3 scripts/run_dixon_coles.py

Writes results/dixon_coles.json (meta + headline + details), results/dixon_coles_by_season.csv,
results/dixon_coles_scorelines.csv, figures/fig3_goals_envelope.png|pdf and
figures/fig9_dixon_coles.png|pdf.  Reads only data/processed/matches.csv.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import figures as F  # noqa: E402
from src.bootstrap import bootstrap_metrics  # noqa: E402
from src.data import CORE_SEASONS, load_matches  # noqa: E402
from src.dixon_coles import (bootstrap_rho, dc_outcome_probs, fig3_goals_envelope, fig9_dixon_coles,  # noqa: E402
                             fit_dixon_coles, goal_histogram_envelope, lr_test, pearson_dispersion,
                             profile_ci, profile_loglik, scoreline_null_envelope, scoreline_table)
from src.evaluate import log_loss, outcome_to_index  # noqa: E402
from src.model import Spec, fit_poisson, make_design, outcome_probs, rates  # noqa: E402
from src.report import frozen_split_forecast, frozen_split_rates  # noqa: E402

SEED = 115
N_BOOT = 1000          # rho bootstrap replicates (pooled and per season); plan asked for >= 200
N_BOOT_EVAL = 2000     # paired bootstrap of log-loss (no refit), same as run_all.py
N_SIM = 2000           # parametric-bootstrap data sets for the scoreline / histogram envelopes
RHO_GRID = np.round(np.arange(-0.20, 0.1001, 0.005), 4)
TRAIN, TEST = "2023-24", "2024-25"
RESULTS = os.path.join(ROOT, "results")


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


def pct_ci(s):
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def main():
    t0 = time.time()
    F.setup_style()
    df = load_matches(CORE_SEASONS)
    core = df[df["season"].isin(CORE_SEASONS)].reset_index(drop=True)
    headline, details = {}, {}

    # ------------------------------------------------------------ per season
    print("[1/6] per-season Dixon–Coles fits and bootstrap")
    rows = []
    for i, s in enumerate(CORE_SEASONS):
        g = core[core["season"] == s]
        d = make_design(g)
        sp = Spec(d)
        fp = fit_poisson(d, spec=sp)
        fd = fit_dixon_coles(d, spec=sp, poisson_fit=fp)
        assert fd.converged and fd.admissible[0] < fd.rho < fd.admissible[1], s
        rng = np.random.default_rng(SEED + 200 + i)
        b = bootstrap_rho(d, fd, N_BOOT, rng, spec=sp)
        lr = lr_test(fd.loglik, fp.loglik)
        lam_p, mu_p = rates(fp.log_alpha, fp.log_beta, fp.log_gamma, d.home, d.away)
        lam_d, mu_d = fd.rates(d)
        oi = outcome_to_index(g["outcome"])
        ll_p = log_loss(outcome_probs(lam_p, mu_p), oi)
        ll_d = log_loss(dc_outcome_probs(lam_d, mu_d, fd.rho), oi)
        disp = pearson_dispersion(d.x, d.y, lam_p, mu_p, fp.n_params)
        dfree = 2 * d.n_matches - fp.n_params
        rows.append({"season": s, "n_matches": d.n_matches, "rho": fd.rho, "rho_lo": b["ci"][0], "rho_hi": b["ci"][1],
                     "rho_se": b["se"], "n_boot": N_BOOT, "n_boot_converged": b["n_converged"], "boot_seed": SEED + 200 + i,
                     "lr_stat": lr["statistic"], "lr_p": lr["p_value"], "loglik_poisson": fp.loglik, "loglik_dc": fd.loglik,
                     "gamma_poisson": fp.gamma, "gamma_dc": fd.gamma,
                     "admissible_lo": fd.admissible[0], "admissible_hi": fd.admissible[1],
                     "insample_logloss_poisson": ll_p, "insample_logloss_dc": ll_d, "insample_logloss_change": ll_d - ll_p,
                     "pearson_dispersion": disp, "dispersion_null_lo": chi2.ppf(0.025, dfree) / dfree,
                     "dispersion_null_hi": chi2.ppf(0.975, dfree) / dfree,
                     "obs_draw_share": float(np.mean(d.x == d.y)),
                     "fitted_draw_share_poisson": float(outcome_probs(lam_p, mu_p)[:, 1].mean()),
                     "fitted_draw_share_dc": float(dc_outcome_probs(lam_d, mu_d, fd.rho)[:, 1].mean())})
        print(f"  {s}: rho = {fd.rho:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]  LR = {lr['statistic']:.2f} "
              f"(p = {lr['p_value']:.3f})  dispersion {disp:.3f}")
    by_season = pd.DataFrame(rows)
    by_season.to_csv(os.path.join(RESULTS, "dixon_coles_by_season.csv"), index=False)
    n_sig = int((by_season["lr_p"] < 0.05).sum())
    n_neg = int((by_season["rho"] < 0).sum())

    # ---------------------------------------------------------------- pooled
    print("[2/6] pooled fit (season-specific strengths, one gamma, one rho), profile, bootstrap")
    dp = make_design(core, season="season")
    spp = Spec(dp)
    fpp = fit_poisson(dp, spec=spp)
    fdp = fit_dixon_coles(dp, spec=spp, poisson_fit=fpp)
    assert fdp.converged and fdp.admissible[0] < fdp.rho < fdp.admissible[1]
    lr_p = lr_test(fdp.loglik, fpp.loglik)
    prof = profile_loglik(dp, RHO_GRID, fdp, spec=spp)
    prof_ci = profile_ci(RHO_GRID, prof)
    strata = pd.factorize(core["season"])[0]
    rng = np.random.default_rng(SEED + 300)
    bp = bootstrap_rho(dp, fdp, N_BOOT, rng, stratify=strata, spec=spp)
    print(f"  pooled rho = {fdp.rho:+.4f}, bootstrap CI {bp['ci']}, profile CI {prof_ci}, LR = {lr_p['statistic']:.2f} "
          f"(p = {lr_p['p_value']:.4f}); admissible range {fdp.admissible}")
    headline["pooled_rho"] = {
        "est": fdp.rho, "ci": bp["ci"], "n": dp.n_matches,
        "how": f"joint MLE over 11 seasons (season-specific alpha, beta; one gamma; one rho; {fdp.n_params} params), "
               f"95% percentile CI from {N_BOOT} season-stratified match-resampling refits (seed {SEED + 300}); "
               f"profile-likelihood CI [{prof_ci[0]:+.4f}, {prof_ci[1]:+.4f}]"}
    headline["pooled_lr_statistic"] = {
        "est": lr_p["statistic"], "ci": [float(chi2.ppf(0.025, 1)), float(chi2.ppf(0.975, 1))], "n": dp.n_matches,
        "how": f"2 x (loglik DC - loglik Poisson) for the pooled fit, 1 df; p = {lr_p['p_value']:.4f}; "
               f"ci = central 95% range of chi2_1 under rho = 0 (not a CI of the statistic)"}
    headline["pooled_loglik_gain_nats"] = {
        "est": fdp.loglik - fpp.loglik, "ci": [0.5 * float(chi2.ppf(0.025, 1)), 0.5 * float(chi2.ppf(0.975, 1))],
        "n": dp.n_matches,
        "how": "scoreline log-likelihood gain of the pooled DC fit over the pooled Poisson fit, total nats; "
               "ci = central 95% range of the gain under rho = 0 (chi2_1 / 2)"}

    # ------------------------------------------------------------ scorelines
    print("[3/6] pooled scoreline table and envelopes")
    lam_p, mu_p = rates(fpp.log_alpha, fpp.log_beta, fpp.log_gamma, dp.home, dp.away)
    lam_d, mu_d = fdp.rates(dp)
    tab = scoreline_table(dp.x, dp.y, lam_p, mu_p, lam_d, mu_d, fdp.rho)
    rng = np.random.default_rng(SEED + 301)
    null_res, null_max = scoreline_null_envelope(lam_p, mu_p, 0.0, N_SIM, rng)
    tab["null_lo_poisson"] = np.percentile(null_res, 2.5, axis=0)
    tab["null_hi_poisson"] = np.percentile(null_res, 97.5, axis=0)
    rng = np.random.default_rng(SEED + 302)
    null_res_dc, null_max_dc = scoreline_null_envelope(lam_d, mu_d, fdp.rho, N_SIM, rng)
    tab.to_csv(os.path.join(RESULTS, "dixon_coles_scorelines.csv"), index=False)
    k = int(np.argmax(np.abs(tab["pearson_poisson"])))
    kd = int(np.argmax(np.abs(tab["pearson_dc"])))
    obs_max, obs_max_dc = float(np.abs(tab["pearson_poisson"]).max()), float(np.abs(tab["pearson_dc"]).max())
    p_max = float(np.mean(null_max >= obs_max))
    p_max_dc = float(np.mean(null_max_dc >= obs_max_dc))
    n_over2 = int((np.abs(tab["pearson_poisson"]) > 2).sum())
    n_over2_dc = int((np.abs(tab["pearson_dc"]) > 2).sum())
    r00 = tab.iloc[0]
    print(tab.round(3).to_string())
    headline["max_abs_pearson_residual_poisson"] = {
        "est": float(tab["pearson_poisson"].iloc[k]), "ci": [float(tab["null_lo_poisson"].iloc[k]), float(tab["null_hi_poisson"].iloc[k])],
        "n": dp.n_matches,
        "how": f"largest |(obs-exp)/sqrt(exp)| of the 17 pooled cells (0..3 x 0..3, 4+) under the pooled Poisson fit, cell "
               f"{tab['scoreline'].iloc[k]}; {n_over2} of 17 cells exceed 2; ci = 95% range of that cell's residual in {N_SIM} "
               f"data sets simulated from the fit (seed {SEED + 301}, no refit); P(max |r| >= observed | model) = {p_max:.3f}"}
    headline["max_abs_pearson_residual_dc"] = {
        "est": float(tab["pearson_dc"].iloc[kd]), "ci": [float(np.percentile(null_res_dc[:, kd], 2.5)), float(np.percentile(null_res_dc[:, kd], 97.5))],
        "n": dp.n_matches,
        "how": f"same under the pooled Dixon-Coles fit, cell {tab['scoreline'].iloc[kd]}; {n_over2_dc} of 17 cells exceed 2; "
               f"ci from {N_SIM} DC simulations (seed {SEED + 302}); P(max |r| >= observed | DC) = {p_max_dc:.3f}"}
    headline["observed_minus_expected_0_0_poisson"] = {
        "est": float(r00["observed"] - r00["expected_poisson"]),
        "ci": [float(r00["null_lo_poisson"] * np.sqrt(r00["expected_poisson"])), float(r00["null_hi_poisson"] * np.sqrt(r00["expected_poisson"]))],
        "n": dp.n_matches,
        "how": f"0-0 count observed {int(r00['observed'])} minus expected {r00['expected_poisson']:.1f} under the pooled Poisson fit "
               f"(DC expects {r00['expected_dc']:.1f}); ci = 95% range of obs - exp in {N_SIM} simulated data sets (seed {SEED + 301})"}

    env = goal_histogram_envelope(dp.x, dp.y, lam_p, mu_p, N_SIM, np.random.default_rng(SEED + 303))
    fig3_path = fig3_goals_envelope(env, title_extra=f", 2015-16..2025-26 (pooled fit; seed {SEED + 303})")
    disp_p = pearson_dispersion(dp.x, dp.y, lam_p, mu_p, fpp.n_params)
    dfree = 2 * dp.n_matches - fpp.n_params
    headline["pearson_dispersion_pooled"] = {
        "est": disp_p, "ci": [float(chi2.ppf(0.025, dfree) / dfree), float(chi2.ppf(0.975, dfree) / dfree)], "n": 2 * dp.n_matches,
        "how": f"sum of squared Pearson residuals of home and away goals / (2m - p), pooled Poisson fit; ci = 95% range under "
               f"the Poisson model (chi2_{dfree}/{dfree}); per-season values {by_season['pearson_dispersion'].min():.3f}.."
               f"{by_season['pearson_dispersion'].max():.3f} in dixon_coles_by_season.csv"}
    n_out = len(env["home"]["outside"]) + len(env["away"]["outside"])
    headline["goal_histogram_bins_outside_band"] = {
        "est": float(n_out), "ci": [0.0, float(2 * len(env["k"]))], "n": dp.n_matches,
        "how": f"bins (0..6+, home and away, 14 total) where the observed count lies outside the 95% parametric-bootstrap band "
               f"({N_SIM} simulated data sets from the pooled Poisson fit, seed {SEED + 303}); ci field = possible range"}

    # ------------------------------------------------------ in-sample log-loss
    print("[4/6] in-sample outcome log-loss")
    oi = outcome_to_index(core["outcome"])
    sets_in = {"poisson": outcome_probs(lam_p, mu_p), "dc": dc_outcome_probs(lam_d, mu_d, fdp.rho)}
    ll_in = {k_: log_loss(p, oi) for k_, p in sets_in.items()}
    b_in = bootstrap_metrics(sets_in, oi, log_loss, n_boot=N_BOOT_EVAL, seed=SEED)
    d_in = b_in["dc"]["samples"] - b_in["poisson"]["samples"]
    m = len(oi)
    headline["insample_logloss_change_per_match"] = {
        "est": ll_in["dc"] - ll_in["poisson"], "ci": pct_ci(d_in), "n": m,
        "how": f"outcome (H/D/A) log-loss of the pooled DC fit minus the pooled Poisson fit over all {m} matches, nats per match "
               f"(negative = DC better); paired bootstrap of matches, {N_BOOT_EVAL} resamples, seed {SEED}, no refit"}
    headline["insample_logloss_change_total_nats"] = {
        "est": m * (ll_in["dc"] - ll_in["poisson"]), "ci": [m * c for c in pct_ci(d_in)], "n": m,
        "how": "the per-match change times the number of matches (total nats over 11 seasons); same bootstrap"}

    # ------------------------------------------------------- held-out 2024-25
    print("[5/6] held-out 2023-24 -> 2024-25")
    fs = frozen_split_forecast(core, TRAIN, TEST)
    oih, hold = fs["outcome_idx"], fs["hold"]
    d23 = fs["train_design"]
    fd23 = fit_dixon_coles(d23, poisson_fit=fs["train_fit"])
    lam_h, mu_h, _, _ = frozen_split_rates(fd23, hold)
    # rho from the seasons before the test season only (2015-16..2023-24), strengths from 2023-24 Poisson
    prior = core[core["season"] < TEST]
    dpr = make_design(prior, season="season")
    fdpr = fit_dixon_coles(dpr)
    sets_h = dict(fs["probs"])
    sets_h["dc_2023_24"] = dc_outcome_probs(lam_h, mu_h, fd23.rho)
    sets_h["dc_prior_pooled_rho"] = dc_outcome_probs(fs["lam"], fs["mu"], fdpr.rho)
    ll_h = {k_: log_loss(p, oih) for k_, p in sets_h.items()}
    b_h = bootstrap_metrics(sets_h, oih, log_loss, n_boot=N_BOOT_EVAL, seed=SEED)
    nh = len(oih)
    with open(os.path.join(RESULTS, "numbers.json")) as fh:
        numbers = json.load(fh)
    pois_ref = numbers["preliminary"]["log_loss"]["poisson"]
    assert abs(ll_h["poisson"] - pois_ref) < 1e-10, (ll_h["poisson"], pois_ref)
    headline["heldout_logloss_poisson"] = {
        "est": ll_h["poisson"], "ci": b_h["poisson"]["ci"], "n": nh,
        "how": f"2023-24 Poisson strengths frozen, promoted = mean of relegated, normalised 0..10 grid (reproduces numbers.json "
               f"preliminary.log_loss.poisson); paired bootstrap of the {nh} test matches, {N_BOOT_EVAL} resamples, seed {SEED}"}
    headline["heldout_logloss_dc"] = {
        "est": ll_h["dc_2023_24"], "ci": b_h["dc_2023_24"]["ci"], "n": nh,
        "how": f"Dixon-Coles fitted jointly on 2023-24 only (rho = {fd23.rho:+.4f}), same freezing and promoted rule; same paired bootstrap"}
    for key, other, lab in (("heldout_logloss_dc_minus_poisson", "dc_2023_24", "DC fitted on 2023-24"),
                            ("heldout_logloss_dc_prior_rho_minus_poisson", "dc_prior_pooled_rho",
                             f"Poisson 2023-24 strengths with rho pooled over 2015-16..2023-24 (rho = {fdpr.rho:+.4f})")):
        s_ = b_h[other]["samples"] - b_h["poisson"]["samples"]
        headline[key] = {"est": ll_h[other] - ll_h["poisson"], "ci": pct_ci(s_), "n": nh,
                         "how": f"paired log-loss difference, {lab} minus Poisson, nats per match on 2024-25 (negative = DC "
                                f"better); {N_BOOT_EVAL} paired resamples, seed {SEED}"}
    s_ = b_h["dc_2023_24"]["samples"] - b_h["pinnacle"]["samples"]
    headline["heldout_logloss_dc_minus_pinnacle"] = {
        "est": ll_h["dc_2023_24"] - ll_h["pinnacle"], "ci": pct_ci(s_), "n": nh,
        "how": "paired difference DC (2023-24) minus Pinnacle pre-match (Fri/Tue collection) on 2024-25; same bootstrap"}
    for k_ in sets_h:
        print(f"  {k_:22s} {ll_h[k_]:.4f} {b_h[k_]['ci']}")

    # ---------------------------------------------------------------- figure 9
    print("[6/6] figures and output")
    fig9_path = fig9_dixon_coles(by_season, {"est": fdp.rho, "ci": bp["ci"]}, RHO_GRID, prof, tab)

    details = {
        "model": "P(x, y) = tau(x, y; lam, mu, rho) Pois(x; lam) Pois(y; mu); tau = 1 - lam mu rho (0-0), 1 + lam rho (0-1), "
                 "1 + mu rho (1-0), 1 - rho (1-1), 1 otherwise; admissible max(-1/lam, -1/mu) <= rho <= min(1/(lam mu), 1)",
        "fit": "joint L-BFGS-B over Poisson log-parameters and rho with analytic gradient, warm-started at the Poisson MLE, rho box [-0.5, 0.5]",
        "pooled": {"rho": fdp.rho, "rho_bootstrap_ci": bp["ci"], "rho_bootstrap_se": bp["se"], "rho_profile_ci": prof_ci,
                   "n_boot_converged": bp["n_converged"], "gamma_poisson": fpp.gamma, "gamma_dc": fdp.gamma,
                   "loglik_poisson": fpp.loglik, "loglik_dc": fdp.loglik, "lr": lr_p, "admissible_range": list(fdp.admissible),
                   "n_params_dc": fdp.n_params, "n_matches": dp.n_matches,
                   "profile": {"rho_grid": RHO_GRID, "loglik": prof}},
        "scorelines": {"n_cells_abs_pearson_over_2_poisson": n_over2, "n_cells_abs_pearson_over_2_dc": n_over2_dc,
                       "p_max_abs_residual_poisson": p_max, "p_max_abs_residual_dc": p_max_dc,
                       "observed_0_0": int(r00["observed"]), "expected_0_0_poisson": float(r00["expected_poisson"]),
                       "expected_0_0_dc": float(r00["expected_dc"]),
                       "observed_1_1": int(tab.iloc[5]["observed"]), "expected_1_1_poisson": float(tab.iloc[5]["expected_poisson"]),
                       "expected_1_1_dc": float(tab.iloc[5]["expected_dc"]),
                       "chi2_pearson_poisson": float((tab["pearson_poisson"] ** 2).sum()),
                       "chi2_pearson_dc": float((tab["pearson_dc"] ** 2).sum())},
        "goal_envelope": {"home_bins_outside": env["home"]["outside"], "away_bins_outside": env["away"]["outside"]},
        "insample_logloss": ll_in,
        "heldout": {"train": TRAIN, "test": TEST, "rho_2023_24": fd23.rho, "rho_prior_pooled": fdpr.rho,
                    "log_loss": ll_h, "log_loss_ci": {k_: v["ci"] for k_, v in b_h.items()},
                    "promoted": fs["promoted"], "relegated": fs["relegated"]},
        "per_season_summary": {"n_rho_negative": n_neg, "n_lr_p_below_0_05": n_sig},
        "seeds": {"per_season_bootstrap": f"{SEED} + 200 + season index", "pooled_bootstrap": SEED + 300,
                  "scoreline_null_poisson": SEED + 301, "scoreline_null_dc": SEED + 302, "goal_envelope": SEED + 303,
                  "paired_logloss_bootstrap": SEED},
        "figures": [os.path.relpath(fig3_path, ROOT), os.path.relpath(fig9_path, ROOT)],
    }
    out = {"meta": {"seed": SEED, "generated": dt.datetime.now().isoformat(timespec="seconds"),
                    "runtime_seconds": None, "n_boot": N_BOOT, "n_boot_eval": N_BOOT_EVAL, "n_sim": N_SIM,
                    "description": "Dixon-Coles rho per season and pooled (season-specific strengths, one gamma, one rho), "
                                   "LR test of rho = 0, pooled 11-season scoreline table with Pearson residuals, goal-histogram "
                                   "envelope, in-sample and held-out (2023-24 -> 2024-25) log-loss with and without the correction."},
           "headline": headline, "details": details}
    out["meta"]["runtime_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, "dixon_coles.json"), "w") as fh:
        json.dump(jsonable(out), fh, indent=2)
    print(f"done in {out['meta']['runtime_seconds']} s")


if __name__ == "__main__":
    main()
