#!/usr/bin/env python3
"""Home advantage over time (workstream B).

Run from the repo root:  python3 scripts/run_timeseries.py
Writes results/timeseries.json, results/timeseries_*.csv, results/crowd_*.csv,
results/coverage_*.csv, figures/fig2_home_advantage_2005_2026.png|pdf,
figures/fig8_crowd_gamma.png|pdf and review/results_timeseries.md.

Does not touch run_all.py outputs.  Every bootstrap or simulation draws from its own
``np.random.default_rng(seed)``; the seeds are recorded in meta.seeds.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.crowd import (CROWD_SEASONS, LEAGUE_NAMES, bootstrap_groups, crowd_design, crowd_status,  # noqa: E402
                       fit_groups, load_league)
from src.data import ALL_SEASONS, CORE_SEASONS, load_matches  # noqa: E402
from src.model import LOG_BOUNDS, Spec, fit_poisson, home_advantage_goals, make_design  # noqa: E402
from src.timeseries import (analytic_log_gamma_se, cluster_weights, cochran_q, cold_vs_warm_recovery,  # noqa: E402
                            coverage_study, date_blocks, dersimonian_laird, fig2_home_advantage_2005_2026,
                            fig8_crowd_status, fit_diagnostics, goals_ratio, iid_weights, lr_common_gamma,
                            pearson_dispersion, percentile_ci, refit_weights, season_z, stratified_iid_weights,
                            trend_meta_regression, trend_z)

SEED = 115
N_BOOT = 2000            # iid bootstrap replicates per season (first 500 reproduce run_all.py's intervals)
N_BOOT_POOLED = 2000     # stratified bootstrap of the pooled common-gamma fits
N_BOOT_CROWD = 2000      # season-stratified bootstrap of the crowd-status fits
N_BOOT_BLOCK = 2000      # date-block and team-cluster bootstraps
N_SIMS_COV, N_BOOT_COV = 500, 200   # coverage study: simulated seasons x bootstrap refits each
N_SIMS_COLD = 200        # cold- vs warm-start recovery
COVID, LOW = "2020-21", "2024-25"
TRUTH_SEASON = "2023-24"
SEEDS = {
    "per_season_iid": "115 + (index in ALL_SEASONS - 10): 2005-06 -> 105 ... 2015-16 -> 115 ... 2025-26 -> 125 "
                      "(identical to run_all.py's per-season seeds for 2015-16..2025-26)",
    "pooled_bootstrap": "reuses the per-season iid replicates: pooled replicate b = replicate b of every season",
    "crowd_bootstrap_E0": SEED + 300, "crowd_bootstrap_D1": SEED + 301, "crowd_bootstrap_SP1": SEED + 302,
    "block_bootstrap": {COVID: SEED + 400, LOW: SEED + 401},
    "team_cluster_bootstrap": {COVID: SEED + 410, LOW: SEED + 411},
    "coverage": SEED + 500, "cold_vs_warm": SEED + 600,
}
RESULTS = os.path.join(ROOT, "results")
REVIEW = os.path.join(ROOT, "review")


def season_seed(s):
    return SEED + ALL_SEASONS.index(s) - 10


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, pd.DataFrame):
        return jsonable(o.reset_index().to_dict(orient="records"))
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def H(est, ci, n, how):
    return {"est": float(est), "ci": [float(ci[0]), float(ci[1])], "n": int(n), "how": how}


def mc_ci_of_width(width, B):
    """Approximate Monte Carlo 95% interval for a bootstrap interval width (relative SD ~ 1/sqrt(2(B-1)))."""
    r = 1.96 / np.sqrt(2 * (B - 1))
    return [width * np.exp(-r), width * np.exp(r)]


def main():
    t0 = time.time()
    df = load_matches()
    assert sorted(df["season"].unique()) == ALL_SEASONS, "matches.csv must hold all 21 seasons"
    out = {"meta": {}, "headline": {}}
    headline = out["headline"]
    diag_max = {"max_abs_theta": 0.0, "max_proj_grad_norm": 0.0, "n_fits": 0, "n_not_converged": 0}

    def track(max_theta, max_pg, n_fits, n_conv):
        diag_max["max_abs_theta"] = max(diag_max["max_abs_theta"], max_theta)
        diag_max["max_proj_grad_norm"] = max(diag_max["max_proj_grad_norm"], max_pg)
        diag_max["n_fits"] += n_fits
        diag_max["n_not_converged"] += n_fits - n_conv

    # ------------------------------------------------------------ 1. per-season fits
    print("[1/7] per-season fits, identity check, iid bootstrap", flush=True)
    fits, designs, boots, W_season, rows = {}, {}, {}, {}, []
    for s in ALL_SEASONS:
        g = df[df["season"] == s]
        d = make_design(g)
        f = fit_poisson(d)
        spec = Spec(d)
        dg = fit_diagnostics(f.theta, d, spec)
        track(dg["max_abs_theta"], dg["proj_grad_norm"], 1, int(f.converged))
        rng = np.random.default_rng(season_seed(s))
        W = iid_weights(d.n_matches, N_BOOT, rng)
        b = refit_weights(d, f, W, spec)
        track(b["max_abs_theta"], b["max_proj_grad_norm"], N_BOOT, b["n_converged"])
        ratio = goals_ratio(d.x, d.y)
        ha = home_advantage_goals(f, d)
        disp = pearson_dispersion(f, d)
        fits[s], designs[s], boots[s], W_season[s] = f, d, b, W
        rows.append({"season": s, "n_matches": d.n_matches, "gamma": f.gamma, "goals_ratio": ratio,
                     "identity_abs_diff": abs(f.gamma - ratio), "gamma_lo": b["gamma_ci"][0], "gamma_hi": b["gamma_ci"][1],
                     "gamma_se": b["gamma_se"], "log_gamma_se_boot": b["log_gamma_se"],
                     "log_gamma_se_analytic": analytic_log_gamma_se(d.x, d.y),
                     "gamma_lo_first500": percentile_ci(b["gamma_samples"][:500])[0],
                     "gamma_hi_first500": percentile_ci(b["gamma_samples"][:500])[1],
                     "ha_goals": ha, "mean_home_minus_away": d.x.mean() - d.y.mean(),
                     "ha_identity_abs_diff": abs(ha - (d.x.mean() - d.y.mean())),
                     "ha_goals_lo": b["ha_goals_ci"][0], "ha_goals_hi": b["ha_goals_ci"][1],
                     "mean_home_goals": d.x.mean(), "mean_away_goals": d.y.mean(),
                     "pearson_dispersion": disp["dispersion"], "dispersion_poisson_lo": disp["poisson_95_range"][0],
                     "dispersion_poisson_hi": disp["poisson_95_range"][1], "loglik": f.loglik, "n_params": f.n_params,
                     "max_abs_theta": dg["max_abs_theta"], "proj_grad_norm": dg["proj_grad_norm"],
                     "boot_max_abs_theta": b["max_abs_theta"], "boot_max_proj_grad_norm": b["max_proj_grad_norm"],
                     "n_boot": N_BOOT, "n_boot_converged": b["n_converged"], "seed": season_seed(s)})
        print(f"  {s}: γ̂ = {f.gamma:.4f} [{b['gamma_ci'][0]:.3f}, {b['gamma_ci'][1]:.3f}]  ΣH/ΣA = {ratio:.6f}  "
              f"dispersion {disp['dispersion']:.3f}", flush=True)
    tab = pd.DataFrame(rows).set_index("season")
    tab.to_csv(os.path.join(RESULTS, "timeseries_by_season.csv"))
    ident = float(tab["identity_abs_diff"].max())
    ident_ha = float(tab["ha_identity_abs_diff"].max())
    assert ident < 1e-6, f"identity gamma_hat = sum H / sum A fails: {ident}"
    # reproduction of run_all.py's 500-replicate intervals (same seeds, same random stream)
    old = pd.read_csv(os.path.join(RESULTS, "home_advantage_by_season.csv")).set_index("season")
    repro = float(np.max(np.abs(np.r_[tab.loc[CORE_SEASONS, "gamma_lo_first500"] - old.loc[CORE_SEASONS, "gamma_lo"],
                                      tab.loc[CORE_SEASONS, "gamma_hi_first500"] - old.loc[CORE_SEASONS, "gamma_hi"]])))
    print(f"  identity max |γ̂ − ΣH/ΣA| = {ident:.2e}; extra goals identity {ident_ha:.2e}; "
          f"first-500 CIs vs run_all.py max diff {repro:.2e}")
    y_log = np.log(tab["gamma"].to_numpy())
    se_log = tab["log_gamma_se_boot"].to_numpy()
    disp_all = float(np.sum([pearson_dispersion(fits[s], designs[s])["pearson_x2"] for s in ALL_SEASONS])
                     / np.sum([pearson_dispersion(fits[s], designs[s])["df"] for s in ALL_SEASONS]))

    # ----------------------------------------------- 2. common-gamma tests and pooled fits
    print("[2/7] common-gamma tests (Cochran's Q, LR, DerSimonian-Laird) and pooled fits", flush=True)
    sets = {
        "all_21_seasons": ALL_SEASONS,
        "crowd_20_seasons": [s for s in ALL_SEASONS if s != COVID],
        "core_11_seasons": CORE_SEASONS,
        "core_crowd_10_seasons": [s for s in CORE_SEASONS if s != COVID],
    }
    tests, pooled = {}, {}
    for name, ss in sets.items():
        idx = [ALL_SEASONS.index(s) for s in ss]
        sub = df[df["season"].isin(ss)]
        dp = make_design(sub, season="season")
        fp = fit_poisson(dp)
        assert fp.converged
        spec_p = Spec(dp)
        dgp = fit_diagnostics(fp.theta, dp, spec_p)
        track(dgp["max_abs_theta"], dgp["proj_grad_norm"], 1, int(fp.converged))
        Wp = np.concatenate([W_season[s][:N_BOOT_POOLED] for s in ss], axis=1)
        bp = refit_weights(dp, fp, Wp, spec_p)
        track(bp["max_abs_theta"], bp["max_proj_grad_norm"], N_BOOT_POOLED, bp["n_converged"])
        ratio = goals_ratio(sub["home_goals"], sub["away_goals"])
        cq = cochran_q(y_log[idx], se_log[idx])
        dl = dersimonian_laird(y_log[idx], se_log[idx])
        lr = lr_common_gamma([fits[s].loglik for s in ss], fp.loglik, len(ss))
        tr = trend_meta_regression(y_log[idx], se_log[idx], [int(s[:4]) for s in ss])
        pred = [float(np.exp(dl["re_mean"] - 1.96 * np.sqrt(dl["tau2"] + dl["re_se"] ** 2))),
                float(np.exp(dl["re_mean"] + 1.96 * np.sqrt(dl["tau2"] + dl["re_se"] ** 2)))]
        pooled[name] = {"seasons": ss, "n_matches": dp.n_matches, "gamma": fp.gamma, "goals_ratio": ratio,
                        "identity_abs_diff": abs(fp.gamma - ratio), "gamma_ci": bp["gamma_ci"],
                        "gamma_se": bp["gamma_se"], "ha_goals": home_advantage_goals(fp, dp),
                        "ha_goals_ci": bp["ha_goals_ci"], "loglik": fp.loglik, "n_params": fp.n_params,
                        "n_boot": N_BOOT_POOLED, "n_boot_converged": bp["n_converged"],
                        "label": "common-gamma interval: assumes one gamma for every season in the set"}
        tests[name] = {"k": len(ss), "cochran_Q": cq["Q"], "Q_df": cq["df"], "Q_p": cq["p"], "I2": cq["I2"],
                       "Q_95_range_under_homogeneity": cq["q_95_range_under_homogeneity"],
                       "LR": lr["LR"], "LR_df": lr["df"], "LR_p": lr["p"],
                       "fe_gamma": float(np.exp(cq["fe_mean"])), "fe_gamma_ci": [float(np.exp(cq["fe_mean"] - 1.96 * cq["fe_se"])),
                                                                                  float(np.exp(cq["fe_mean"] + 1.96 * cq["fe_se"]))],
                       "dl_tau": dl["tau"], "dl_tau_ci": dl["tau_ci_qprofile"], "re_gamma": float(np.exp(dl["re_mean"])),
                       "re_gamma_ci": [float(np.exp(dl["re_mean"] - 1.96 * dl["re_se"])),
                                       float(np.exp(dl["re_mean"] + 1.96 * dl["re_se"]))],
                       "re_prediction_interval_one_season": pred,
                       "pooled_mle_gamma": fp.gamma, "pooled_mle_gamma_ci": bp["gamma_ci"],
                       "trend_slope_log_gamma_per_season": tr["slope_log_gamma_per_season"],
                       "trend_slope_ci": tr["slope_ci"], "trend_slope_p": tr["slope_p"],
                       "trend_residual_Q": tr["residual_Q"], "trend_residual_df": tr["residual_df"],
                       "trend_residual_p": tr["residual_p"]}
        print(f"  {name}: Q = {cq['Q']:.2f} on {cq['df']} df (p = {cq['p']:.4f}); LR = {lr['LR']:.2f} (p = {lr['p']:.4f}); "
              f"τ = {dl['tau']:.4f}; pooled γ = {fp.gamma:.4f} {bp['gamma_ci']}  (ΣH/ΣA = {ratio:.6f})", flush=True)
    pd.DataFrame(tests).T.to_csv(os.path.join(RESULTS, "timeseries_common_gamma_tests.csv"))
    ident_pooled = max(p["identity_abs_diff"] for p in pooled.values())

    # z of 2020-21 and 2024-25 against the other seasons
    zrows = []
    for set_name, ss in [("all_21_seasons", ALL_SEASONS), ("core_11_seasons", CORE_SEASONS)]:
        for target in (COVID, LOW):
            ref = [s for s in ss if s not in (COVID, target)]
            i = ALL_SEASONS.index(target)
            ri = [ALL_SEASONS.index(s) for s in ref]
            z = season_z(y_log[i], se_log[i], y_log[ri], se_log[ri])
            z |= trend_z(y_log[i], se_log[i], int(target[:4]), y_log[ri], se_log[ri], [int(s[:4]) for s in ref])
            zrows.append({"set": set_name, "season": target, "gamma": float(np.exp(y_log[i])),
                          "reference": f"{len(ref)} seasons of {set_name} excluding 2020-21" +
                                       ("" if target == COVID else " and 2024-25")} | z)
    ztab = pd.DataFrame(zrows)
    ztab.to_csv(os.path.join(RESULTS, "timeseries_season_z.csv"), index=False)
    print(ztab[["set", "season", "z_common", "z_random", "ref_tau"]].round(3).to_string(index=False))

    # ----------------------------------------------------------------- 3. crowd status
    print("[3/7] crowd-status fits, Premier League 2019-20 .. 2021-22", flush=True)
    cdf = df[df["season"].isin(CROWD_SEASONS)].reset_index(drop=True)
    status = crowd_status(cdf, "E0")
    status_date_only = crowd_status(cdf, "E0", dec2020_by_club=False)
    status_split = status.copy()
    status_split[(status == "closed") & (cdf["season"] == "2019-20")] = "closed_restart_2019_20"
    flags = cdf[["season", "date", "home", "away", "home_goals", "away_goals"]].copy()
    flags["crowd_status"] = status
    flags["crowd_status_dec2020_date_only"] = status_date_only
    flags.to_csv(os.path.join(RESULTS, "crowd_flags_e0.csv"), index=False)
    counts = pd.crosstab(cdf["season"], status)
    print(counts.to_string())
    strata = pd.factorize(cdf["season"])[0]
    Wc = stratified_iid_weights(strata, N_BOOT_CROWD, np.random.default_rng(SEEDS["crowd_bootstrap_E0"]))

    def crowd_fit(st, labels, W):
        d, gidx, present = crowd_design(cdf, st, labels)
        f = fit_groups(d, gidx, present)
        track(f.max_abs_theta, f.proj_grad_norm, 1, int(f.converged))
        b = bootstrap_groups(d, gidx, present, f, W)
        track(b["max_abs_theta"], b["max_proj_grad_norm"], len(W), b["n_converged"])
        return d, gidx, present, f, b

    def summarise(present, f, b, st, frame, ref="full"):
        res = {"groups": {}, "differences": {}}
        for k, lab in enumerate(present):
            m = (st == lab).to_numpy()
            res["groups"][lab] = {"gamma": float(np.exp(f.log_gamma[k])), "gamma_ci": percentile_ci(b["gamma_samples"][:, k]),
                                  "n_matches": int(m.sum()),
                                  "raw_goals_ratio": goals_ratio(frame["home_goals"][m], frame["away_goals"][m])}
        j = present.index(ref)
        for k, lab in enumerate(present):
            if lab != ref:
                diff = b["gamma_samples"][:, k] - b["gamma_samples"][:, j]
                res["differences"][f"{lab}_minus_{ref}"] = {
                    "est": float(np.exp(f.log_gamma[k]) - np.exp(f.log_gamma[j])), "ci": percentile_ci(diff),
                    "share_boot_below_0": float(np.mean(diff < 0))}
        res.update({"loglik": f.loglik, "converged": f.converged, "n_boot": len(b["gamma_samples"]),
                    "n_boot_converged": b["n_converged"]})
        return res

    crowd = {}
    _, _, pr, f, b = crowd_fit(status, ("full", "closed", "limited"), Wc)
    crowd["E0_main"] = summarise(pr, f, b, status, cdf)
    _, _, pr, f, b = crowd_fit(status_date_only, ("full", "closed", "limited"), Wc)
    crowd["E0_dec2020_date_only"] = summarise(pr, f, b, status_date_only, cdf)
    _, _, pr, f, b = crowd_fit(status_split, ("full", "closed_restart_2019_20", "closed", "limited"), Wc)
    crowd["E0_restart_split"] = summarise(pr, f, b, status_split, cdf)
    # closed vs fans-allowed (full + limited merged), the definition used for D1/SP1
    st2 = status.replace({"limited": "full"})
    _, _, pr, f, b = crowd_fit(st2, ("full", "closed"), Wc)
    crowd["E0_two_status"] = summarise(pr, f, b, st2, cdf)
    crowd["E0_counts_by_season"] = {s: {k: int(v) for k, v in r.items()} for s, r in counts.iterrows()}
    for k, v in crowd["E0_main"]["groups"].items():
        print(f"  E0 {k:8s} γ = {v['gamma']:.3f} {np.round(v['gamma_ci'], 3)}  n = {v['n_matches']}")
    for k, v in crowd["E0_main"]["differences"].items():
        print(f"  E0 {k}: {v['est']:+.3f} {np.round(v['ci'], 3)}")
    print(f"  restart split: {crowd['E0_restart_split']['differences']}")

    # ---------------------------------------------------------- 4. block bootstraps
    print("[4/7] date-block and team-cluster bootstraps", flush=True)
    block = {}
    brows = []
    for s in (COVID, LOW):
        g = df[df["season"] == s].reset_index(drop=True)
        d, f = designs[s], fits[s]
        blocks = date_blocks(g["date"], 10)
        assert len(np.unique(blocks)) == 38 and np.all(np.bincount(blocks) == 10)
        Wb = cluster_weights(blocks, N_BOOT_BLOCK, np.random.default_rng(SEEDS["block_bootstrap"][s]))
        Wt = cluster_weights(g["home"].to_numpy(), N_BOOT_BLOCK, np.random.default_rng(SEEDS["team_cluster_bootstrap"][s]))
        bb = refit_weights(d, f, Wb)
        bt = refit_weights(d, f, Wt)
        track(bb["max_abs_theta"], bb["max_proj_grad_norm"], N_BOOT_BLOCK, bb["n_converged"])
        track(bt["max_abs_theta"], bt["max_proj_grad_norm"], N_BOOT_BLOCK, bt["n_converged"])
        bi = boots[s]
        res = {}
        for lab, bx in (("iid", bi), ("date_block_38x10", bb), ("team_cluster_home", bt)):
            w = bx["gamma_ci"][1] - bx["gamma_ci"][0]
            res[lab] = {"gamma_ci": bx["gamma_ci"], "width": w, "width_mc_ci": mc_ci_of_width(w, bx["n_boot"]),
                        "gamma_se": bx["gamma_se"], "log_gamma_se": bx["log_gamma_se"], "n_boot": bx["n_boot"],
                        "n_converged": bx["n_converged"]}
            brows.append({"season": s, "scheme": lab, "gamma": f.gamma, "lo": bx["gamma_ci"][0], "hi": bx["gamma_ci"][1],
                          "width": w, "se": bx["gamma_se"], "width_ratio_to_iid": w / (bi["gamma_ci"][1] - bi["gamma_ci"][0]),
                          "n_boot": bx["n_boot"]})
        res["analytic_wald_width"] = float(f.gamma * (np.exp(1.96 * analytic_log_gamma_se(d.x, d.y))
                                                       - np.exp(-1.96 * analytic_log_gamma_se(d.x, d.y))))
        block[s] = res
        print(f"  {s}: widths iid {res['iid']['width']:.3f}, block {res['date_block_38x10']['width']:.3f}, "
              f"team {res['team_cluster_home']['width']:.3f}, analytic {res['analytic_wald_width']:.3f}", flush=True)
    pd.DataFrame(brows).to_csv(os.path.join(RESULTS, "timeseries_block_bootstrap.csv"), index=False)

    # ---------------------------------------------------- 5. coverage and cold start
    print(f"[5/7] CI coverage: {N_SIMS_COV} simulated {TRUTH_SEASON} seasons x {N_BOOT_COV} bootstrap refits", flush=True)
    tc = time.time()
    cov = coverage_study(designs[TRUTH_SEASON], fits[TRUTH_SEASON], N_SIMS_COV, N_BOOT_COV,
                         np.random.default_rng(SEEDS["coverage"]))
    per_sim = cov.pop("per_sim")
    per_sim.to_csv(os.path.join(RESULTS, "coverage_per_sim.csv"), index=False)
    track(float(per_sim["max_abs_theta"].max()), float(per_sim["max_proj_grad_norm"].max()),
          N_SIMS_COV * (N_BOOT_COV + 1), int(per_sim["n_converged"].sum()))
    cov["runtime_seconds"] = time.time() - tc
    cov["mean_boot_se_log"] = float(per_sim["boot_se_log"].mean())
    cov["mean_analytic_se_log"] = float(per_sim["analytic_se_log"].mean())
    cov["sd_log_gamma_hat_across_sims"] = float(np.log(per_sim["gamma_hat"]).std(ddof=1))
    cov["max_abs_gamma_minus_goals_ratio"] = float(np.abs(per_sim["gamma_hat"] - per_sim["goals_ratio"]).max())
    cov_rows = [{"interval": k, **cov[f"coverage_{k}"]} for k in ("percentile", "wald_log", "analytic")]
    pd.DataFrame([{"interval": r["interval"], "share": r["share"], "n_covered": r["n_covered"], "ci_lo": r["ci"][0],
                   "ci_hi": r["ci"][1], "n_sims": N_SIMS_COV, "n_boot": N_BOOT_COV} for r in cov_rows]).to_csv(
        os.path.join(RESULTS, "coverage_summary.csv"), index=False)
    print(f"  coverage percentile {cov['coverage_percentile']}, wald-log {cov['coverage_wald_log']}, "
          f"analytic {cov['coverage_analytic']}  ({cov['runtime_seconds']:.0f} s)", flush=True)
    cw = cold_vs_warm_recovery(designs[TRUTH_SEASON], fits[TRUTH_SEASON], N_SIMS_COLD,
                               np.random.default_rng(SEEDS["cold_vs_warm"]))
    track(cw["max_abs_theta"], cw["max_proj_grad_norm"], cw["n_fits"], cw["n_converged"])
    print(f"  cold vs warm: bias warm {cw['warm']['gamma_bias']:+.4f}, cold {cw['cold']['gamma_bias']:+.4f}, "
          f"max |Δγ̂| {cw['max_abs_gamma_diff_cold_vs_warm']:.2e}", flush=True)

    # --------------------------------------------------------- 6. other leagues
    print("[6/7] Bundesliga and La Liga 2019-20 .. 2021-22 (stretch)", flush=True)
    other = {}
    for L, seed_key in (("D1", "crowd_bootstrap_D1"), ("SP1", "crowd_bootstrap_SP1")):
        ldf = load_league(L)
        per = {}
        for s in CROWD_SEASONS:
            g = ldf[ldf["season"] == s]
            d = make_design(g)
            f = fit_poisson(d)
            dg = fit_diagnostics(f.theta, d, Spec(d))
            track(dg["max_abs_theta"], dg["proj_grad_norm"], 1, int(f.converged))
            per[s] = {"n_matches": d.n_matches, "n_teams": d.n_teams, "gamma": f.gamma,
                      "goals_ratio": goals_ratio(d.x, d.y), "identity_abs_diff": abs(f.gamma - goals_ratio(d.x, d.y)),
                      "analytic_gamma_ci": [f.gamma * float(np.exp(-1.96 * analytic_log_gamma_se(d.x, d.y))),
                                            f.gamma * float(np.exp(1.96 * analytic_log_gamma_se(d.x, d.y)))]}
        st = crowd_status(ldf, L)
        Wl = stratified_iid_weights(pd.factorize(ldf["season"])[0], N_BOOT_CROWD, np.random.default_rng(SEEDS[seed_key]))
        d, gidx, present = crowd_design(ldf, st, ("full", "closed"))
        f = fit_groups(d, gidx, present)
        track(f.max_abs_theta, f.proj_grad_norm, 1, int(f.converged))
        b = bootstrap_groups(d, gidx, present, f, Wl)
        track(b["max_abs_theta"], b["max_proj_grad_norm"], len(Wl), b["n_converged"])
        other[L] = summarise(present, f, b, st, ldf) | {"per_season": per, "league": LEAGUE_NAMES[L],
                                                        "counts_by_season": {s: {k: int(v) for k, v in r.items()}
                                                                             for s, r in pd.crosstab(ldf["season"], st).iterrows()}}
        print(f"  {L}: {other[L]['groups']}  diff {other[L]['differences']}", flush=True)
    crowd["other_leagues"] = other
    crow = []
    for lg, res in [("E0", crowd["E0_main"]), ("E0_two_status", crowd["E0_two_status"]),
                    ("E0_dec2020_date_only", crowd["E0_dec2020_date_only"]), ("E0_restart_split", crowd["E0_restart_split"]),
                    ("D1", other["D1"]), ("SP1", other["SP1"])]:
        for k, v in res["groups"].items():
            crow.append({"fit": lg, "status": k, "gamma": v["gamma"], "lo": v["gamma_ci"][0], "hi": v["gamma_ci"][1],
                         "n_matches": v["n_matches"], "raw_goals_ratio": v["raw_goals_ratio"]})
        for k, v in res["differences"].items():
            crow.append({"fit": lg, "status": k, "gamma": v["est"], "lo": v["ci"][0], "hi": v["ci"][1]})
    pd.DataFrame(crow).to_csv(os.path.join(RESULTS, "crowd_gamma_by_status.csv"), index=False)

    # ------------------------------------------------------------------ 7. figures
    print("[7/7] figures", flush=True)
    t20 = tests["crowd_20_seasons"]
    p20 = pooled["crowd_20_seasons"]
    fig2_home_advantage_2005_2026(
        tab, {"est": p20["gamma"], "lo": p20["gamma_ci"][0], "hi": p20["gamma_ci"][1],
              "label": f"common γ, 20 seasons excl. 2020-21: {p20['gamma']:.3f} [{p20['gamma_ci'][0]:.3f}, {p20['gamma_ci'][1]:.3f}]"},
        {"lo": t20["re_prediction_interval_one_season"][0], "hi": t20["re_prediction_interval_one_season"][1],
         "label": f"random-effects 95% range for one season (τ = {t20['dl_tau']:.3f})"}, n_boot=N_BOOT)
    frows = []
    for L, res in (("E0", crowd["E0_main"]), ("D1", other["D1"]), ("SP1", other["SP1"])):
        for k, v in res["groups"].items():
            frows.append({"league": L, "status": k, "est": v["gamma"], "lo": v["gamma_ci"][0], "hi": v["gamma_ci"][1],
                          "n": v["n_matches"]})
    fdiffs = []
    for lab, v in (("Premier League\n(closed − full)", crowd["E0_main"]["differences"]["closed_minus_full"]),
                   ("PL 2019-20 restart\n(92 matches) − full", crowd["E0_restart_split"]["differences"]["closed_restart_2019_20_minus_full"]),
                   ("PL 2020-21 closed\n− full", crowd["E0_restart_split"]["differences"]["closed_minus_full"]),
                   ("Bundesliga\n(closed − fans allowed)", other["D1"]["differences"]["closed_minus_full"]),
                   ("La Liga\n(closed − fans allowed)", other["SP1"]["differences"]["closed_minus_full"])):
        fdiffs.append({"label": lab, "est": v["est"], "lo": v["ci"][0], "hi": v["ci"][1]})
    fig8_crowd_status(frows, fdiffs)

    # ------------------------------------------------------------------ headline
    g20, g24 = tab.loc[COVID], tab.loc[LOW]
    zr = {(r["set"], r["season"]): r for r in zrows}
    nb = N_BOOT
    headline["common_gamma_20_crowd_seasons"] = H(p20["gamma"], p20["gamma_ci"], p20["n_matches"],
        f"pooled MLE, season-specific strengths and one shared gamma over the 20 seasons 2005-06..2025-26 except 2020-21 "
        f"(= sum H / sum A); common-gamma interval: percentile CI from {N_BOOT_POOLED} season-stratified bootstrap refits")
    headline["common_gamma_21_seasons"] = H(pooled["all_21_seasons"]["gamma"], pooled["all_21_seasons"]["gamma_ci"],
        pooled["all_21_seasons"]["n_matches"], "as common_gamma_20_crowd_seasons but all 21 seasons; common-gamma interval")
    headline["gamma_2020_21"] = H(g20["gamma"], [g20["gamma_lo"], g20["gamma_hi"]], 380,
        f"per-season MLE (= sum H / sum A); percentile CI from {nb} iid match-bootstrap refits, seed {season_seed(COVID)}")
    headline["gamma_2024_25"] = H(g24["gamma"], [g24["gamma_lo"], g24["gamma_hi"]], 380,
        f"per-season MLE (= sum H / sum A); percentile CI from {nb} iid match-bootstrap refits, seed {season_seed(LOW)}")
    headline["extra_home_goals_2020_21"] = H(g20["ha_goals"], [g20["ha_goals_lo"], g20["ha_goals_hi"]], 380,
        f"mean(lambda - lambda/gamma) = mean H - mean A (identity); percentile CI from {nb} iid bootstrap refits")
    headline["extra_home_goals_2024_25"] = H(g24["ha_goals"], [g24["ha_goals_lo"], g24["ha_goals_hi"]], 380,
        f"mean(lambda - lambda/gamma) = mean H - mean A (identity); percentile CI from {nb} iid bootstrap refits")
    headline["extra_home_goals_common_20_crowd_seasons"] = H(p20["ha_goals"], p20["ha_goals_ci"], p20["n_matches"],
        "pooled common-gamma fit, mean(lambda - lambda/gamma); season-stratified bootstrap percentile CI")
    headline["identity_max_abs_gamma_minus_goals_ratio"] = H(ident, [float(tab["identity_abs_diff"].min()), ident], 21,
        "max over 21 seasons of |gamma_hat (L-BFGS-B) - sum home goals / sum away goals|; 'ci' is [min, max] over seasons, "
        f"not a confidence interval; pooled fits: max {ident_pooled:.1e}; extra-goals identity: max {ident_ha:.1e}")
    for nm, key in (("cochran_q_21_seasons", "all_21_seasons"), ("cochran_q_20_crowd_seasons", "crowd_20_seasons"),
                    ("cochran_q_11_core_seasons", "core_11_seasons"), ("cochran_q_10_core_crowd_seasons", "core_crowd_10_seasons")):
        t = tests[key]
        headline[nm] = H(t["cochran_Q"], t["Q_95_range_under_homogeneity"], t["Q_df"],
            f"Cochran's Q on log gamma_hat with bootstrap SEs ({nb} reps), df = n = {t['Q_df']}, p = {t['Q_p']:.4f}; "
            f"'ci' is the central 95% range of chi2_{t['Q_df']} under a common gamma")
        headline[nm.replace("cochran_q", "lr_common_gamma")] = H(t["LR"], tests[key]["Q_95_range_under_homogeneity"], t["LR_df"],
            f"2 x (sum of per-season log-likelihoods - pooled common-gamma log-likelihood), df = n = {t['LR_df']}, "
            f"p = {t['LR_p']:.4f}; 'ci' is the central 95% range of chi2_{t['LR_df']} under H0")
    for nm, key in (("trend_log_gamma_per_season_20_crowd_seasons", "crowd_20_seasons"),
                    ("trend_log_gamma_per_season_10_core_crowd_seasons", "core_crowd_10_seasons")):
        t = tests[key]
        headline[nm] = H(t["trend_slope_log_gamma_per_season"], t["trend_slope_ci"], t["k"],
            f"inverse-variance weighted regression of log gamma_hat on season start year (bootstrap SEs), SE inflated by "
            f"sqrt(residual Q/df) if > 1; p = {t['trend_slope_p']:.2g}; residual Q = {t['trend_residual_Q']:.1f} on "
            f"{t['trend_residual_df']} df (p = {t['trend_residual_p']:.3f})")
    for nm, key in (("dl_tau_20_crowd_seasons", "crowd_20_seasons"), ("dl_tau_21_seasons", "all_21_seasons"),
                    ("dl_tau_10_core_crowd_seasons", "core_crowd_10_seasons")):
        t = tests[key]
        headline[nm] = H(t["dl_tau"], t["dl_tau_ci"], t["k"],
            "DerSimonian-Laird between-season SD of log gamma; 95% CI by the Q-profile method (Viechtbauer 2007)")
    for season, short in ((COVID, "2020_21"), (LOW, "2024_25")):
        r = zr[("all_21_seasons", season)]
        headline[f"z_{short}_common_gamma"] = H(r["z_common"], [r["z_common"] - 1.96, r["z_common"] + 1.96], r["n_ref"],
            f"(log gamma_s - fixed-effect mean of {r['n_ref']} other crowd seasons) / sqrt(se_s^2 + se_mean^2), bootstrap SEs; "
            f"p = {r['p_common_two_sided']:.4f}; 'ci' = z +- 1.96; ratio to common gamma {r['ratio_to_common']:.3f} "
            f"[{r['ratio_to_common_ci'][0]:.3f}, {r['ratio_to_common_ci'][1]:.3f}]")
        headline[f"z_{short}_random_effects"] = H(r["z_random"], [r["z_random"] - 1.96, r["z_random"] + 1.96], r["n_ref"],
            f"(log gamma_s - DL random-effects mean of {r['n_ref']} other crowd seasons) / sqrt(se_s^2 + tau^2 + se_mean^2), "
            f"tau = {r['ref_tau']:.4f}; p = {r['p_random_two_sided']:.4f}; 'ci' = z +- 1.96")
        headline[f"z_{short}_vs_trend"] = H(r["z_trend"], [r["z_trend"] - 1.96, r["z_trend"] + 1.96], r["n_ref"],
            f"(log gamma_s - linear-trend prediction for that season from the {r['n_ref']} other crowd seasons) / "
            f"sqrt(se_s^2 + var(prediction)); trend-predicted gamma {r['trend_predicted_gamma']:.3f}; "
            f"p = {r['p_trend_two_sided']:.4f}; 'ci' = z +- 1.96")
    headline["pearson_dispersion_pooled_21_seasons"] = H(disp_all, [float(tab["pearson_dispersion"].min()),
                                                                     float(tab["pearson_dispersion"].max())], 21,
        f"sum of Pearson X^2 over home and away goals / sum of (760 - 40) over 21 per-season fits; 'ci' is the [min, max] "
        f"of the per-season dispersions; a Poisson season lies in [{tab['dispersion_poisson_lo'].iloc[0]:.3f}, "
        f"{tab['dispersion_poisson_hi'].iloc[0]:.3f}] with 95% probability (chi2_720/720)")
    m = crowd["E0_main"]
    for lab in m["groups"]:
        v = m["groups"][lab]
        headline[f"crowd_gamma_{lab}_e0"] = H(v["gamma"], v["gamma_ci"], v["n_matches"],
            f"crowd-status Poisson fit, season-specific strengths 2019-20..2021-22, one gamma per status; percentile CI "
            f"from {N_BOOT_CROWD} season-stratified bootstrap refits; status flagged by date (attendance not in data)")
    for k, v in m["differences"].items():
        headline[f"crowd_gamma_{k}_e0"] = H(v["est"], v["ci"], m["groups"][k.split("_minus_")[0]]["n_matches"],
            f"difference of the two gammas in the same crowd-status fit; percentile CI of the paired bootstrap differences "
            f"({N_BOOT_CROWD} season-stratified refits)")
    rs = crowd["E0_restart_split"]["differences"]["closed_restart_2019_20_minus_full"]
    headline["crowd_gamma_restart_2019_20_minus_full_e0"] = H(rs["est"], rs["ci"], 92,
        "gamma(92 restart matches of 2019-20) - gamma(full crowds) in a 4-status fit; paired bootstrap percentile CI")
    for L in ("D1", "SP1"):
        v = other[L]["differences"]["closed_minus_full"]
        headline[f"crowd_gamma_closed_minus_fans_allowed_{L.lower()}"] = H(v["est"], v["ci"],
            other[L]["groups"]["closed"]["n_matches"],
            f"{LEAGUE_NAMES[L]} 2019-20..2021-22, closed = restart to end of 2020-21, partial crowds not separated; "
            f"paired season-stratified bootstrap ({N_BOOT_CROWD}) percentile CI")
    for s, short in ((COVID, "2020_21"), (LOW, "2024_25")):
        for lab in ("iid", "date_block_38x10", "team_cluster_home"):
            v = block[s][lab]
            headline[f"ci_width_{lab}_{short}"] = H(v["width"], v["width_mc_ci"], v["n_boot"],
                f"width of the 95% percentile interval of gamma, {lab} bootstrap ({v['n_boot']} refits); 'ci' is the approx "
                f"Monte Carlo interval width*exp(+-1.96/sqrt(2(B-1)))")
    for k, lab in (("percentile", "bootstrap percentile"), ("wald_log", "Wald on log gamma with bootstrap SE"),
                   ("analytic", "analytic log(H/A) +- 1.96 sqrt(1/H + 1/A)")):
        v = cov[f"coverage_{k}"]
        headline[f"ci_coverage_{k}"] = H(v["share"], v["ci"], N_SIMS_COV,
            f"share of {N_SIMS_COV} simulated {TRUTH_SEASON}-style seasons (truth gamma = {cov['true_gamma']:.4f}) whose 95% "
            f"{lab} interval ({N_BOOT_COV} refits) contains the truth; Clopper-Pearson 95% CI")
    headline["recovery_gamma_bias_warm_start"] = H(cw["warm"]["gamma_bias"], cw["warm"]["bias_ci"], N_SIMS_COLD,
        f"mean(gamma_hat) - true gamma over {N_SIMS_COLD} simulated {TRUTH_SEASON} seasons, fit started at the truth; "
        "CI = bias +- 1.96 Monte Carlo SE")
    headline["recovery_gamma_bias_cold_start"] = H(cw["cold"]["gamma_bias"], cw["cold"]["bias_ci"], N_SIMS_COLD,
        f"same simulated seasons, fit started at the default x0 (strengths 0, gamma = mean H / mean A); "
        f"max |gamma_cold - gamma_warm| = {cw['max_abs_gamma_diff_cold_vs_warm']:.1e}")
    headline["max_abs_theta_all_fits"] = H(diag_max["max_abs_theta"], [0.0, diag_max["max_abs_theta"]], diag_max["n_fits"],
        f"largest |log-parameter| over every fit in this script (n = number of fits); bound is +-{LOG_BOUNDS[1]:.0f}; "
        f"'ci' is [0, max]; no parameter within {LOG_BOUNDS[1] - diag_max['max_abs_theta']:.2f} of a bound")
    headline["max_projected_gradient_norm_all_fits"] = H(diag_max["max_proj_grad_norm"], [0.0, diag_max["max_proj_grad_norm"]],
        diag_max["n_fits"], "largest infinity-norm of the box-projected gradient of the negative log-likelihood at the "
        "returned optimum, over every fit; 'ci' is [0, max]")

    # ------------------------------------------------------------------ write
    out["meta"] = {"seed": SEED, "seeds": SEEDS, "generated": dt.datetime.now().isoformat(timespec="seconds"),
                   "runtime_seconds": None, "n_boot": N_BOOT, "n_boot_pooled": N_BOOT_POOLED, "n_boot_crowd": N_BOOT_CROWD,
                   "n_boot_block": N_BOOT_BLOCK, "n_sims_coverage": N_SIMS_COV, "n_boot_coverage": N_BOOT_COV,
                   "n_sims_cold_vs_warm": N_SIMS_COLD, "seasons": ALL_SEASONS,
                   "description": "Workstream B: Premier League home advantage 2005-06..2025-26. Per-season Poisson gamma "
                                  "with bootstrap CIs and the goals-ratio identity check; common-gamma tests; crowd-status "
                                  "fits 2019-20..2021-22 (E0, D1, SP1); date-block and team-cluster bootstraps; CI coverage "
                                  "and cold-start recovery; fit diagnostics. Attendance is not in the data: crowd status is "
                                  "flagged from match dates.",
                   "identity": "For a double round robin with geometric-mean(alpha)=1, gamma_hat = sum home goals / sum away "
                               "goals exactly, and extra home goals per match = mean H - mean A.",
                   "fit_diagnostics": diag_max}
    out["per_season"] = tab
    out["pooled_common_gamma"] = pooled
    out["common_gamma_tests"] = tests
    out["season_z"] = zrows
    out["crowd"] = crowd
    out["block_bootstrap"] = block
    out["coverage"] = cov
    out["cold_vs_warm"] = cw
    out["verification"] = {"identity_max_abs_diff_per_season": ident, "identity_max_abs_diff_pooled": ident_pooled,
                           "extra_goals_identity_max_abs_diff": ident_ha,
                           "identity_max_abs_diff_other_leagues": max(v["identity_abs_diff"] for L in other.values()
                                                                      for v in L["per_season"].values()),
                           "first500_ci_vs_run_all_max_abs_diff": repro,
                           "coverage_sims_max_abs_gamma_minus_goals_ratio": cov["max_abs_gamma_minus_goals_ratio"]}
    out["meta"]["runtime_seconds"] = time.time() - t0
    with open(os.path.join(RESULTS, "timeseries.json"), "w") as fh:
        json.dump(jsonable(out), fh, indent=1)
    write_review()
    print(f"done in {out['meta']['runtime_seconds']:.0f} s")


def _ci(c, nd=3):
    return f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


def write_review(json_path=os.path.join(RESULTS, "timeseries.json"), md_path=os.path.join(REVIEW, "results_timeseries.md")):
    """Human-readable results note, generated from results/timeseries.json (re-run with --review-only)."""
    with open(json_path) as fh:
        J = json.load(fh)
    h, meta, T, P = J["headline"], J["meta"], J["common_gamma_tests"], J["pooled_common_gamma"]
    ps = {r["season"]: r for r in J["per_season"]}
    Z = {(r["set"], r["season"]): r for r in J["season_z"]}
    cr, oth, blk, cov, cw = J["crowd"], J["crowd"]["other_leagues"], J["block_bootstrap"], J["coverage"], J["cold_vs_warm"]
    fd = meta["fit_diagnostics"]

    def hv(k, nd=3):
        return f"{h[k]['est']:.{nd}f} {_ci(h[k]['ci'], nd)}"

    L = []
    a = L.append
    a("# Results: home advantage over time (workstream B)\n")
    a(f"Generated {meta['generated']} by `python3 scripts/run_timeseries.py` in {meta['runtime_seconds']:.0f} s "
      f"(seed {meta['seed']}; per-component seeds in `results/timeseries.json` → `meta.seeds`). Premier League "
      f"2005-06 to 2025-26, 21 seasons × 380 matches = 7,980 matches. Every number has a 95% interval and the way it "
      f"was obtained. `results/timeseries.json` → `headline` has the machine-readable version of each.\n")
    a("Rubric sections served: **Verification (8)** in §1 and §5, **Estimation and uncertainty (10)** in §1, §2, §4 and §5, "
      "**Assumptions (8)** in §3 (crowd status equals season label; independent matches) and §4, and "
      "**What this does not show (4)** in §3 and §6.\n")

    a("## 1. γ per season and the goals-ratio identity (Verification, Model, Estimation)\n")
    a("**Analytic solution the code is checked against.** For one season of a double round robin with "
      "geometric-mean(α) = 1, the γ score equation gives Σ(x − λ) = 0, and summing the α and β score equations gives "
      "Σ(x − λ) + Σ(y − μ) = 0, so Σλ = ΣH and Σμ = ΣA. Every ordered pair (i, j) is played once, so "
      "Σ α_h β_a = Σ α_a β_h, and therefore\n")
    a("    γ̂ = Σ home goals / Σ away goals   (exactly),    extra home goals per match = mean(λ − λ/γ̂) = mean H − mean A.\n")
    a("The same argument with season-specific strengths and one shared γ gives the pooled common-γ MLE = ΣH/ΣA over "
      "the pooled seasons. The 40-parameter model adds nothing to the point estimate of γ. It matters for the forecasts "
      "and for the bootstrap: a resample is not a balanced schedule, so each replicate needs a refit.\n")
    v = J["verification"]
    a(f"- Identity check, L-BFGS-B γ̂ against ΣH/ΣA, all 21 seasons: max |difference| = "
      f"**{v['identity_max_abs_diff_per_season']:.1e}** (tolerance 1e-6). Pooled common-γ fits: "
      f"{v['identity_max_abs_diff_pooled']:.1e}. Extra-goals identity: {v['extra_goals_identity_max_abs_diff']:.1e}. "
      f"Bundesliga and La Liga seasons (18 and 20 teams): {v['identity_max_abs_diff_other_leagues']:.1e}. In 500 "
      f"simulated seasons: {v['coverage_sims_max_abs_gamma_minus_goals_ratio']:.1e}.")
    a(f"- Reproduction: with the same seeds, the first 500 of the 2,000 bootstrap replicates reproduce `run_all.py`'s "
      f"500-replicate intervals for 2015-16..2025-26 to {v['first500_ci_vs_run_all_max_abs_diff']:.1e}.")
    a(f"- Intervals: percentile, {meta['n_boot']} iid match-bootstrap refits per season (warm-started at the season fit; "
      f"seed 115 + season index − 10).\n")
    a("| Season | γ̂ = ΣH/ΣA | 95% CI | extra home goals | 95% CI | Pearson dispersion |")
    a("|---|---|---|---|---|---|")
    for s in meta["seasons"]:
        r = ps[s]
        a(f"| {s} | {r['gamma']:.3f} | {_ci([r['gamma_lo'], r['gamma_hi']])} | {r['ha_goals']:.3f} | "
          f"{_ci([r['ha_goals_lo'], r['ha_goals_hi']])} | {r['pearson_dispersion']:.3f} |")
    lo_d, hi_d = ps[meta["seasons"][0]]["dispersion_poisson_lo"], ps[meta["seasons"][0]]["dispersion_poisson_hi"]
    outside = [s for s in meta["seasons"] if not lo_d <= ps[s]["pearson_dispersion"] <= hi_d]
    a("")
    a(f"Pearson dispersion after the fit (X² over home and away goals / (760 − 40)) ranges "
      f"{h['pearson_dispersion_pooled_21_seasons']['ci'][0]:.2f} to {h['pearson_dispersion_pooled_21_seasons']['ci'][1]:.2f} "
      f"(pooled over 21 seasons: {h['pearson_dispersion_pooled_21_seasons']['est']:.3f}). Under the Poisson model a season "
      f"falls in [{lo_d:.3f}, {hi_d:.3f}] with 95% probability (χ²₇₂₀/720). "
      + (f"Seasons outside that range: {', '.join(outside)}, all *below* it, which means under-dispersion relative to the fit, not "
         f"over-dispersion. " if outside and all(ps[s]['pearson_dispersion'] < lo_d for s in outside) else
         (f"Seasons outside that range: {', '.join(outside)}. " if outside else "No season falls outside it. "))
      + "The earlier figure of 0.88–1.09 over 11 seasons is replaced by these 21-season numbers. There is no sign of "
        "over-dispersion, which supports the Poisson variance assumption for the bootstrap and the tests below.\n")

    a("## 2. One γ for all seasons? Cochran's Q, likelihood ratio, DerSimonian–Laird (Estimation, Assumptions)\n")
    a("Inputs: y_s = log γ̂_s with the bootstrap SD of log γ̂ as its SE. Q is the inverse-variance Cochran statistic. LR is "
      "2 × (Σ per-season log-likelihoods − pooled common-γ log-likelihood), the pooled fit having season-specific "
      "strengths and one γ. τ is the DerSimonian–Laird between-season SD of log γ, with a Q-profile CI. The trend is an "
      "inverse-variance weighted regression of log γ̂ on season start year.\n")
    a("| Seasons | k | Cochran Q (df, p) | LR (df, p) | DL τ [95% CI] | common γ (pooled MLE) [95% CI] | trend in log γ per season [95% CI] |")
    a("|---|---|---|---|---|---|---|")
    names = {"all_21_seasons": "all 21 (2005-06..2025-26)", "crowd_20_seasons": "20 crowd seasons (excl. 2020-21)",
             "core_11_seasons": "11 core (2015-16..2025-26)", "core_crowd_10_seasons": "10 core crowd seasons (excl. 2020-21)"}
    for k, nm in names.items():
        t, p = T[k], P[k]
        a(f"| {nm} | {t['k']} | {t['cochran_Q']:.1f} ({t['Q_df']}, {t['Q_p']:.4f}) | {t['LR']:.1f} ({t['LR_df']}, {t['LR_p']:.4f}) | "
          f"{t['dl_tau']:.3f} {_ci(t['dl_tau_ci'])} | {p['gamma']:.3f} {_ci(p['gamma_ci'])} | "
          f"{t['trend_slope_log_gamma_per_season']:+.4f} {_ci(t['trend_slope_ci'], 4)} |")
    a("")
    a("Every pooled interval is a **common-γ interval**: it assumes one γ for all seasons in the set. It comes from "
      f"{meta['n_boot_pooled']} season-stratified bootstrap refits, where pooled replicate b uses replicate b of every season.\n")
    t20, t10, t21, t11 = T["crowd_20_seasons"], T["core_crowd_10_seasons"], T["all_21_seasons"], T["core_11_seasons"]
    a("**Reconciliation with `review/final_plan_by_rubric.md` §1 row 3, with the actual numbers.** Over the 11 core seasons "
      f"the plan's sentence holds. Among the 10 core crowd seasons there is no evidence of between-season variation "
      f"(Q = {t10['cochran_Q']:.1f} on {t10['Q_df']} df, p = {t10['Q_p']:.2f}; LR = {t10['LR']:.1f}, p = {t10['LR_p']:.2f}). "
      f"Adding 2020-21 rejects a common γ (Q = {t11['cochran_Q']:.1f} on {t11['Q_df']} df, p = {t11['Q_p']:.3f}; "
      f"LR = {t11['LR']:.1f}, p = {t11['LR_p']:.3f}; the LR reproduces the earlier 22.0 on 10 df). Alexia's Q of 14.3 "
      f"(p = 0.11) used different SEs (500-replicate SD of γ̂ rather than 2,000-replicate SD of log γ̂); here it is "
      f"{t10['cochran_Q']:.1f}. **Over all 21 seasons the sentence does not hold.** The 20 crowd seasons already reject a "
      f"common γ (Q = {t20['cochran_Q']:.1f} on {t20['Q_df']} df, p = {t20['Q_p']:.4f}; LR p = {t20['LR_p']:.3f}). Home "
      f"advantage has declined since 2005-06: the slope is {t20['trend_slope_log_gamma_per_season']:+.4f} "
      f"{_ci(t20['trend_slope_ci'], 4)} in log γ per season, about "
      f"{100 * (np.exp(10 * t20['trend_slope_log_gamma_per_season']) - 1):+.0f}% in γ per decade, with p = "
      f"{t20['trend_slope_p']:.1g}. The residual heterogeneity around the trend is Q = {t20['trend_residual_Q']:.1f} on "
      f"{t20['trend_residual_df']} df (p = {t20['trend_residual_p']:.2f}). Proposed wording for the write-up: *\"Over "
      f"2015-16 to 2025-26 we find no evidence of between-season variation among the 10 crowd seasons, and adding "
      f"2020-21 rejects a common γ. Over 2005-06 to 2025-26 home advantage declines by about "
      f"{abs(100 * (np.exp(10 * t20['trend_slope_log_gamma_per_season']) - 1)):.0f}% per decade, so a single γ is "
      f"rejected even without 2020-21.\"* The common γ over the 20 crowd seasons ({hv('common_gamma_20_crowd_seasons')}) "
      f"is therefore an average over a declining series, not a constant.\n")

    a("### 2020-21 and 2024-25 against the other seasons\n")
    a("Three reference models, all on log γ with bootstrap SEs. *Common γ*: z = (y_s − fixed-effect mean) / "
      "√(se_s² + se_mean²). *Random effects*: z = (y_s − DL mean) / √(se_s² + τ² + se_mean²), with τ estimated on the "
      "reference seasons. *Trend*: z = (y_s − linear-trend prediction for that season) / √(se_s² + var(prediction)). "
      "The reference set always excludes 2020-21 and the season being tested.\n")
    a("| Set | Season | γ̂ | z common γ (p) | z random effects (τ) | z vs trend (trend-predicted γ) | γ̂ / common γ [95% CI] |")
    a("|---|---|---|---|---|---|---|")
    for st in ("all_21_seasons", "core_11_seasons"):
        for s in ("2020-21", "2024-25"):
            r = Z[(st, s)]
            a(f"| {names[st]} | {s} | {r['gamma']:.3f} | {r['z_common']:.2f} ({r['p_common_two_sided']:.4f}) | "
              f"{r['z_random']:.2f} ({r['ref_tau']:.3f}) | {r['z_trend']:.2f} ({r['trend_predicted_gamma']:.3f}) | "
              f"{r['ratio_to_common']:.3f} {_ci(r['ratio_to_common_ci'])} |")
    a("")
    za, zb = Z[("all_21_seasons", "2020-21")], Z[("all_21_seasons", "2024-25")]
    a(f"**2024-25 needs addressing directly.** It played in full stadiums, and its γ̂ = {ps['2024-25']['gamma']:.3f} "
      f"{_ci([ps['2024-25']['gamma_lo'], ps['2024-25']['gamma_hi']])} is nearly as low as 2020-21's "
      f"{ps['2020-21']['gamma']:.3f} {_ci([ps['2020-21']['gamma_lo'], ps['2020-21']['gamma_hi']])}. Under every reference "
      f"model both seasons sit more than 2 SDs below the others, and they are about equally unusual except against the "
      f"trend, where 2024-25 is less extreme. Against all other crowd seasons the z values (2020-21 vs 2024-25) are "
      f"{za['z_common']:.2f} vs {zb['z_common']:.2f} under a common γ, {za['z_random']:.2f} vs {zb['z_random']:.2f} under "
      f"random effects, and {za['z_trend']:.2f} vs {zb['z_trend']:.2f} against the trend. The earlier rough figures were "
      f"−3.3/−3.0 and −2.5/−2.3; the 11-season rows above are their recomputation. A single low season is therefore not "
      f"evidence of an empty-stadium effect: a season as low as 2020-21 happened with full crowds four years later. "
      f"The crowd evidence has to come from within-season contrasts (§3) and other leagues (§6), and those are weak.\n")

    a("## 3. Crowd status by match date, Premier League 2019-20 to 2021-22 (Assumptions, What this does not show)\n")
    a("**Attendance is not in the football-data files.** Crowd status is a flag from the match date (and, for "
      "December 2020, the home club), not a head count:\n")
    c = cr["E0_counts_by_season"]
    a(f"- *closed*: from the 2019-20 restart on 17 June 2020 ({c['2019-20'].get('closed', 0)} matches) to the end of "
      f"2020-21 ({c['2020-21'].get('closed', 0)} matches), apart from the limited matches below;")
    a(f"- *limited* ({sum(v.get('limited', 0) for v in c.values())} matches): 2–15 December 2020 with the home club in "
      "a tier-2 area (London, Liverpool City Region, Brighton & Hove, Hampshire: Arsenal, Chelsea, Crystal Palace, "
      "Fulham, Tottenham, West Ham, Liverpool, Everton, Brighton, Southampton; up to 2,000 fans), and 18–23 May 2021 "
      "(last two rounds, all grounds, up to 10,000 fans). **The club list comes from the tier allocation of 2 December "
      "2020 and was not verified against attendance records.** Crystal Palace and Fulham are added to the brief's "
      "list because they are London clubs under the same rule. A date-only sensitivity flag treats all 21 matches of "
      "2–15 Dec as limited;")
    a("- *full*: everything else (2019-20 before the restart and all of 2021-22).\n")
    a(f"Fit: season-specific α and β, one γ per crowd status. Percentile CIs come from {meta['n_boot_crowd']} "
      "season-stratified bootstrap refits; differences use the paired replicates of the same fit.\n")
    a("| Fit | Status | n | γ [95% CI] | raw ΣH/ΣA |")
    a("|---|---|---|---|---|")
    for fit_name in ("E0_main", "E0_dec2020_date_only", "E0_restart_split", "E0_two_status"):
        for k, g in cr[fit_name]["groups"].items():
            a(f"| {fit_name} | {k} | {g['n_matches']} | {g['gamma']:.3f} {_ci(g['gamma_ci'])} | {g['raw_goals_ratio']:.3f} |")
    a("")
    a("| Fit | Difference | est [95% CI] | share of replicates < 0 |")
    a("|---|---|---|---|")
    for fit_name in ("E0_main", "E0_dec2020_date_only", "E0_restart_split", "E0_two_status"):
        for k, d in cr[fit_name]["differences"].items():
            a(f"| {fit_name} | {k} | {d['est']:+.3f} {_ci(d['ci'])} | {d['share_boot_below_0']:.3f} |")
    a("")
    m = cr["E0_main"]["differences"]["closed_minus_full"]
    rs = cr["E0_restart_split"]["differences"]
    a(f"**Reading.** Closed doors minus full crowds: {m['est']:+.3f} {_ci(m['ci'])}. As expected the interval is wide, "
      f"and it includes 0. The 92 restart matches of 2019-20 do not show lower home advantage: restart − full = "
      f"{rs['closed_restart_2019_20_minus_full']['est']:+.3f} {_ci(rs['closed_restart_2019_20_minus_full']['ci'])}. The "
      f"closed matches of 2020-21 do: {rs['closed_minus_full']['est']:+.3f} {_ci(rs['closed_minus_full']['ci'])}. The "
      f"limited-crowd γ rests on 30 matches and is uninformative. So the treatment 'crowd status = season label' breaks "
      f"in both directions: 2019-20 contains 92 empty-stadium matches with ordinary home advantage, and the low value "
      f"is specific to 2020-21 rather than to empty stadiums as such. That fits a 2020-21 season effect (which could be "
      f"crowds, fixture congestion, or chance; see 2024-25 above) as well as it fits a crowd effect.\n")

    a("## 4. Date-block and team-cluster bootstraps (Estimation, Assumptions: independent matches)\n")
    a("Blocks: the season's 380 matches sorted by date and cut into 38 consecutive blocks of 10. The 38 blocks are "
      "resampled with replacement. Team cluster: the 20 home teams are resampled with replacement, each bringing all "
      f"19 of its home matches. Each scheme uses {meta['n_boot_block']} refits. The analytic Wald width uses "
      "SE(log γ̂) = √(1/ΣH + 1/ΣA).\n")
    a("| Season | iid | date blocks 38×10 | team cluster (home) | analytic Wald |")
    a("|---|---|---|---|---|")
    for s in ("2020-21", "2024-25"):
        b = blk[s]
        a(f"| {s} | {b['iid']['width']:.3f}: {_ci(b['iid']['gamma_ci'])} | {b['date_block_38x10']['width']:.3f}: "
          f"{_ci(b['date_block_38x10']['gamma_ci'])} | {b['team_cluster_home']['width']:.3f}: "
          f"{_ci(b['team_cluster_home']['gamma_ci'])} | {b['analytic_wald_width']:.3f} |")
    a("")
    b1, b2 = blk["2020-21"], blk["2024-25"]
    a(f"Entries are the 95% interval width followed by the interval. The Monte Carlo error of a 2,000-replicate width is "
      f"about ±3%. Results, as they came out: in 2020-21 the date-block interval is *narrower* than the iid one "
      f"({b1['date_block_38x10']['width'] / b1['iid']['width']:.2f}×). In 2024-25 it is wider "
      f"({b2['date_block_38x10']['width'] / b2['iid']['width']:.2f}×). The team-cluster interval is wider in both seasons "
      f"({b1['team_cluster_home']['width'] / b1['iid']['width']:.2f}× and "
      f"{b2['team_cluster_home']['width'] / b2['iid']['width']:.2f}×). With only 38 blocks or 20 clusters these widths are "
      f"themselves noisy, and a team-cluster resample perturbs the balance of the schedule more than an iid resample "
      f"does. Read this as: *the iid interval could be understated by up to about "
      f"{100 * (max(b1['team_cluster_home']['width'] / b1['iid']['width'], b2['team_cluster_home']['width'] / b2['iid']['width']) - 1):.0f}% "
      f"if home-team clustering matters, but there is no consistent sign of within-season time dependence.* None of the "
      f"schemes moves 2020-21's or 2024-25's interval enough to change a conclusion above. The 2020-21 interval contains "
      f"1 under every scheme.\n")

    a("## 5. CI coverage, cold start, and fit diagnostics (Verification, Estimation)\n")
    a(f"Truth = the 2023-24 fit (γ = {cov['true_gamma']:.4f}). {cov['n_sims']} simulated seasons were drawn on the real "
      f"fixture list (seed {meta['seeds']['coverage']}). Each was fitted, then refitted on {cov['n_boot']} iid bootstrap "
      f"resamples. Coverage is the share of 95% intervals that contain the true γ, with a Clopper–Pearson CI.\n")
    a("| Interval | coverage [95% CI] |")
    a("|---|---|")
    for k, lab in (("percentile", "bootstrap percentile (the one used in Fig 2)"), ("wald_log", "Wald on log γ, bootstrap SE"),
                   ("analytic", "analytic log(ΣH/ΣA) ± 1.96 √(1/ΣH + 1/ΣA)")):
        x = cov[f"coverage_{k}"]
        a(f"| {lab} | {x['share']:.3f} {_ci(x['ci'])} ({x['n_covered']}/{cov['n_sims']}) |")
    a("")
    a(f"Nominal 95% is inside every coverage CI, so the bootstrap error bars in Fig 2 have the coverage they claim when "
      f"the model is true. Mean bootstrap SE(log γ̂) = {cov['mean_boot_se_log']:.4f}, mean analytic SE = "
      f"{cov['mean_analytic_se_log']:.4f}, and the SD of log γ̂ across simulations is "
      f"{cov['sd_log_gamma_hat_across_sims']:.4f}. Coverage under a misspecified data-generating process (over-dispersion, "
      f"correlated goals, drifting strengths) is not tested.\n")
    a(f"**Cold vs warm start** ({cw['n_sims']} simulated 2023-24 seasons, seed {meta['seeds']['cold_vs_warm']}, each "
      f"fitted from the truth and from the default x0): γ̂ bias warm {hv('recovery_gamma_bias_warm_start', 4)}, cold "
      f"{hv('recovery_gamma_bias_cold_start', 4)} (bias ± 1.96 Monte Carlo SE). Max |γ̂_cold − γ̂_warm| = "
      f"{cw['max_abs_gamma_diff_cold_vs_warm']:.1e}, and max |θ_cold − θ_warm| = {cw['max_abs_theta_diff_cold_vs_warm']:.1e}. "
      f"The warm start at the truth does not drive the recovery result, because the likelihood is concave and both starts "
      f"reach the same optimum. Mean iterations: {cw['warm']['mean_iter']:.0f} warm, {cw['cold']['mean_iter']:.0f} cold.\n")
    a(f"**Bounds and gradients.** Over all {fd['n_fits']:,} fits in this script (data fits, every bootstrap refit, every "
      f"simulation): max |θ| = {fd['max_abs_theta']:.2f} against bounds of ±5. **No parameter came within "
      f"{5 - fd['max_abs_theta']:.2f} of a bound**, which is more than the 1 required. Max projected-gradient norm "
      f"(∞-norm, in nats per unit log-parameter) = {fd['max_proj_grad_norm']:.1e}. Fits not reporting convergence: "
      f"{fd['n_not_converged']}. The per-season data fits reach at most "
      f"{max(r['proj_grad_norm'] for r in J['per_season']):.1e}, and their bootstrap refits at most "
      f"{max(r['boot_max_proj_grad_norm'] for r in J['per_season']):.1e}. The largest values come from the 400–840-parameter "
      "pooled bootstrap refits, which stop on L-BFGS-B's relative-ftol criterion (1e-13). In a separate check, 60 such "
      "21-season refits had gradients up to 8e-3; polishing them with two further warm-started runs changed γ̂ by at "
      "most 9e-7, so the residual gradient has no effect on any reported digit.\n")

    a("## 6. Bundesliga and La Liga, 2019-20 to 2021-22 (stretch item: a parallel comparison, not a pooled analysis)\n")
    a("Same fit as §3 with two statuses. *closed* = from the restart (Bundesliga 16 May 2020, La Liga 11 June 2020) to "
      "the end of 2020-21. *fans allowed* = everything else. The partial crowds in those leagues are **not** separated, "
      "because the dates were not verified: Bundesliga September–October 2020, capacity caps and renewed closures in "
      "2021-22, and a few La Liga matches at the end of 2020-21. The 'fans allowed' group is therefore a mix, and the "
      "contrast is biased toward 0.\n")
    a("| League | γ fans allowed [95% CI] (n) | γ closed [95% CI] (n) | closed − fans allowed [95% CI] |")
    a("|---|---|---|---|")
    e2 = cr["E0_two_status"]
    for nm, rr in (("Premier League (limited merged into fans allowed)", e2), ("Bundesliga (D1)", oth["D1"]), ("La Liga (SP1)", oth["SP1"])):
        g, d = rr["groups"], rr["differences"]["closed_minus_full"]
        a(f"| {nm} | {g['full']['gamma']:.3f} {_ci(g['full']['gamma_ci'])} ({g['full']['n_matches']}) | "
          f"{g['closed']['gamma']:.3f} {_ci(g['closed']['gamma_ci'])} ({g['closed']['n_matches']}) | {d['est']:+.3f} {_ci(d['ci'])} |")
    a("")
    a("All three point estimates are negative, and only La Liga's interval excludes 0. Per-season γ̂ (= ΣH/ΣA, analytic "
      "Wald CI) for each league is in `timeseries.json` → `crowd.other_leagues.<league>.per_season`. The data identify a "
      "difference between periods, not its cause.\n")

    a("## 7. Figures\n")
    a("- `figures/fig2_home_advantage_2005_2026.png|pdf` replaces fig2 and fig2b, which `run_all.py` still writes, so the "
      "integrator retires those. It shows γ̂ per season with 2,000-replicate CIs; the common-γ band (20 seasons excluding "
      "2020-21, labelled as a common-γ interval); the random-effects 95% range for one season's γ; and the identity in "
      "the title. Extra home goals = mean H − mean A is in `results/timeseries_by_season.csv`.")
    a("- `figures/fig8_crowd_gamma.png|pdf`: γ by crowd status for the Premier League, Bundesliga and La Liga, and the "
      "closed − fans-allowed differences.\n")

    a("## 8. Where this disagrees with `review/final_plan_by_rubric.md`, and what was set aside\n")
    a(f"- §1 row 3: the plan says there is no evidence of between-season variation among the crowd seasons. That holds over "
      f"the 10 core crowd seasons only. Over the 20 crowd seasons since 2005-06 a common γ is rejected (p = {t20['Q_p']:.3f}), "
      "because of a downward trend.")
    a("- The plan expected the block bootstrap to be wider. It is narrower for 2020-21 and wider for 2024-25. The "
      "team-cluster bootstrap is wider in both seasons.")
    a(f"- Replicates were *increased*: 2,000 bootstrap replicates per season instead of 500, and {cov['n_sims']} × "
      f"{cov['n_boot']} coverage simulations instead of 200 × 200. The script still runs in "
      f"{meta['runtime_seconds'] / 60:.1f} min.")
    a("- The plan's Dec 2020 tier-2 list is extended by Crystal Palace and Fulham (London clubs). The list is not verified "
      "against attendance records, and a date-only sensitivity fit is reported.")
    a("- No changes were made to `src/data.py` or `data/team_names.csv`. The other leagues are read through the existing "
      "`load_raw_season(label, league)` in `src/crowd.py`. Their team names stay in the football-data short form and are "
      "not added to the E0 name table.")
    a("- Set aside: coverage under a misspecified data-generating process; a full random-effects or time-series model "
      "for γ (the DL τ and the linear trend are reported instead); verifying the partial-crowd dates for the Bundesliga "
      "and La Liga.")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return md_path


if __name__ == "__main__":
    if "--review-only" in sys.argv:
        print(write_review())
    else:
        main()
