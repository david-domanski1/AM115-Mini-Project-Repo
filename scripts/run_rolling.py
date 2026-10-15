#!/usr/bin/env python3
"""Rolling-origin forecast evaluation, test seasons 2016-17 .. 2025-26 (Agent A).

Run from the repo root:  python3 scripts/run_rolling.py
Writes results/rolling.json, results/rolling_*.csv, figures/fig6_rolling_logloss.*,
figures/fig7_rolling_reliability.*.  Model and estimation are documented in src/rolling.py.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import figures as F  # noqa: E402
from src import rolling as R  # noqa: E402
from src.data import load_matches  # noqa: E402
from src.evaluate import outcome_to_index  # noqa: E402

SEED = 115
N_BOOT_EVAL = 2000       # paired match bootstrap per season (no refitting)
N_BOOT_PARAM = 200       # training-set bootstrap with full rolling refits, 2024-25 only
N_BOOT_BET = 2000
HALF_LIVES = [None, 730, 365, 180]          # days; None = no decay
GAMMA_MODES = ["one", "season"]
TEST_SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2016, 2026)]
PARAM_SEASON = "2024-25"
BET_SEASONS = ["2024-25", "2025-26"]
MARGINS = [0.0, 0.05, 0.10]
Z80 = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)   # 2.80: detectable gap = Z80 * SE
RESULTS = os.path.join(ROOT, "results")


def hl_name(h):
    return "none" if h is None else f"{h}d"


def cfg_name(h, g, rule="relegated"):
    return f"hl={hl_name(h)}|gamma={g}|promoted={rule}"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def pct(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def run_config(L, O, h, g, rule):
    """Rolling forecasts for all test seasons under one configuration -> DataFrame."""
    parts = []
    for s in TEST_SEASONS:
        r = R.rolling_forecast(L, s, h, g, rule)
        d = L.df.iloc[r["idx"]][["season", "date", "home", "away", "home_spell", "away_spell",
                                  "home_goals", "away_goals", "outcome"]].copy()
        d["lam"], d["mu"] = r["lam"], r["mu"]
        d[["pH", "pD", "pA"]] = r["p"]
        d["oi"] = O[r["idx"]]
        d["promoted_match"] = [L.spell_start[a] == s or L.spell_start[b] == s
                               for a, b in zip(d["home_spell"], d["away_spell"])]
        d["n_fail"] = r["n_fail"]
        parts.append(d)
    out = pd.concat(parts)
    out["config"] = cfg_name(h, g, rule)
    return out


def main():
    global N_BOOT_PARAM
    if "--quick" in sys.argv:          # smoke test only: tiny training bootstrap
        N_BOOT_PARAM = 4
    t0 = time.time()
    df = load_matches()
    L = R.make_layout(df)
    O = outcome_to_index(L.df["outcome"].astype(str))
    n_fail_total = 0

    # ------------------------------------------------------------ 1. configuration grid
    print("[1] rolling forecasts, half-life x gamma grid (promoted rule: mean of relegated)")
    preds = {}
    for g in GAMMA_MODES:
        for h in HALF_LIVES:
            t = time.time()
            preds[cfg_name(h, g)] = run_config(L, O, h, g, "relegated")
            print(f"    {cfg_name(h, g):45s} {time.time() - t:5.1f}s  mean log-loss "
                  f"{R.per_match_logloss(preds[cfg_name(h, g)][['pH', 'pD', 'pA']].to_numpy(), preds[cfg_name(h, g)]['oi'].to_numpy()).mean():.4f}")

    def season_ll(p):
        ll = R.per_match_logloss(p[["pH", "pD", "pA"]].to_numpy(), p["oi"].to_numpy())
        return pd.Series(ll, index=p.index).groupby(p["season"].to_numpy()).mean()

    grid = pd.DataFrame({k: season_ll(v) for k, v in preds.items()})
    grid_mean = grid.mean()
    best = grid_mean.idxmin()
    best_h = next(h for h in HALF_LIVES if f"hl={hl_name(h)}|" in best)
    best_g = "season" if "gamma=season" in best else "one"
    first5, last5 = TEST_SEASONS[:5], TEST_SEASONS[5:]
    best_first5 = grid.loc[first5].mean().idxmin()
    best_last5 = grid.loc[last5].mean().idxmin()
    print(f"    best over 10 seasons: {best}; first five: {best_first5}; last five: {best_last5}")

    # paired comparisons between configs (pooled over the 3,800 matches, same match draws)
    rng_cfg = np.random.default_rng(SEED + 4000)
    losses = {k: R.per_match_logloss(v[["pH", "pD", "pA"]].to_numpy(), v["oi"].to_numpy()) for k, v in preds.items()}
    pairs = [(k, best) for k in preds if k != best]
    cfg_boot = R.paired_bootstrap(losses, rng_cfg, N_BOOT_EVAL, pairs=pairs)

    hl_ok = [h for h in HALF_LIVES if cfg_name(h, best_g) == best or cfg_boot[f"{cfg_name(h, best_g)}-{best}"]["ci"][0] <= 0]
    none_ok = None in hl_ok
    hl_ok_range = [min(h for h in hl_ok if h is not None), max(h for h in hl_ok if h is not None)]
    nodecay_key = f"{cfg_name(None, best_g)}-{best}"
    decay_gain = cfg_boot[nodecay_key] if nodecay_key in cfg_boot else {"est": 0.0, "ci": [0.0, 0.0]}

    # ------------------------------------------------------------ 2. promoted-team rules
    print("[2] promoted-team rules at the best half-life and gamma mode")
    for rule in R.PROMOTED_RULES:
        k = cfg_name(best_h, best_g, rule)
        if k not in preds:
            preds[k] = run_config(L, O, best_h, best_g, rule)
    rule_keys = {rule: cfg_name(best_h, best_g, rule) for rule in R.PROMOTED_RULES}
    rng_rule = np.random.default_rng(SEED + 5000)
    rl = {r: R.per_match_logloss(preds[k][["pH", "pD", "pA"]].to_numpy(), preds[k]["oi"].to_numpy())
          for r, k in rule_keys.items()}
    promo_mask = preds[best]["promoted_match"].to_numpy()
    rule_all = R.paired_bootstrap(rl, rng_rule, N_BOOT_EVAL,
                                  pairs=[(a, b) for i, a in enumerate(R.PROMOTED_RULES) for b in R.PROMOTED_RULES[i + 1:]])
    rule_promo = R.paired_bootstrap({r: v[promo_mask] for r, v in rl.items()}, rng_rule, N_BOOT_EVAL,
                                    pairs=[(a, b) for i, a in enumerate(R.PROMOTED_RULES) for b in R.PROMOTED_RULES[i + 1:]])
    rule_means = {r: rule_all[r]["est"] for r in R.PROMOTED_RULES}
    r_best, r_worst = min(rule_means, key=rule_means.get), max(rule_means, key=rule_means.get)
    key_rw = f"{r_worst}-{r_best}" if f"{r_worst}-{r_best}" in rule_all else f"{r_best}-{r_worst}"
    sign = 1 if key_rw == f"{r_worst}-{r_best}" else -1
    rule_range = {"est": sign * rule_all[key_rw]["est"],
                  "ci": sorted([sign * c for c in rule_all[key_rw]["ci"]]),
                  "worst": r_worst, "best": r_best}
    rule_range_promo = {"est": sign * rule_promo[key_rw]["est"],
                        "ci": sorted([sign * c for c in rule_promo[key_rw]["ci"]])}
    for r in R.PROMOTED_RULES:
        print(f"    {r:11s} all {rule_all[r]['est']:.4f}  promoted-team matches {rule_promo[r]['est']:.4f}")

    # ------------------------------------------------------------ 3. scoring per season (primary config)
    print("[3] per-season scoring against base rate and Pinnacle (primary config)")
    P = preds[best].copy()
    P = P.merge(L.df[["date", "home", "away", "p_PS_H", "p_PS_D", "p_PS_A", "p_PSC_H", "p_PSC_D", "p_PSC_A",
                      "PSCH", "PSCD", "PSCA"]], on=["date", "home", "away"], how="left")
    rows, samples = [], {}
    for k, s in enumerate(TEST_SEASONS):
        rng = np.random.default_rng(SEED + 1000 + k)
        d = P[P["season"] == s]
        oi = d["oi"].to_numpy()
        pp = d[["pH", "pD", "pA"]].to_numpy()
        pb = np.tile(R.base_rate_from_prior(df, s), (len(d), 1))
        ps = d[["p_PS_H", "p_PS_D", "p_PS_A"]].to_numpy()
        pc = d[["p_PSC_H", "p_PSC_D", "p_PSC_A"]].to_numpy()
        has = ~np.isnan(ps).any(axis=1) & ~np.isnan(pc).any(axis=1)
        onehot = np.eye(3)[oi]
        vec_all = {"ll_poisson": R.per_match_logloss(pp, oi), "ll_base": R.per_match_logloss(pb, oi),
                   "br_poisson": R.per_match_brier(pp, oi), "br_base": R.per_match_brier(pb, oi),
                   "citl_H": pp[:, 0] - onehot[:, 0], "citl_D": pp[:, 1] - onehot[:, 1], "citl_A": pp[:, 2] - onehot[:, 2],
                   "res_home_goals": d["lam"].to_numpy() - d["home_goals"].to_numpy(),
                   "res_away_goals": d["mu"].to_numpy() - d["away_goals"].to_numpy()}
        b_all = R.paired_bootstrap(vec_all, rng, N_BOOT_EVAL, pairs=[("ll_poisson", "ll_base"), ("br_poisson", "br_base")],
                                   return_samples=True)
        vec_odds = {"ll_poisson_o": R.per_match_logloss(pp[has], oi[has]), "ll_ps": R.per_match_logloss(ps[has], oi[has]),
                    "ll_psc": R.per_match_logloss(pc[has], oi[has]), "ll_base_o": R.per_match_logloss(pb[has], oi[has]),
                    "br_poisson_o": R.per_match_brier(pp[has], oi[has]), "br_ps": R.per_match_brier(ps[has], oi[has]),
                    "br_psc": R.per_match_brier(pc[has], oi[has]),
                    "citl_psc_H": pc[has, 0] - onehot[has, 0], "citl_psc_D": pc[has, 1] - onehot[has, 1],
                    "citl_psc_A": pc[has, 2] - onehot[has, 2]}
        b_odds = R.paired_bootstrap(vec_odds, rng, N_BOOT_EVAL,
                                    pairs=[("ll_poisson_o", "ll_ps"), ("ll_poisson_o", "ll_psc"), ("ll_ps", "ll_base_o"),
                                           ("br_poisson_o", "br_ps"), ("br_poisson_o", "br_psc")],
                                    return_samples=True)
        row = {"season": s, "n": len(d), "n_odds": int(has.sum()),
               "mean_lambda": d["lam"].mean(), "realised_home_goals": d["home_goals"].mean(),
               "mean_mu": d["mu"].mean(), "realised_away_goals": d["away_goals"].mean()}
        for bb in (b_all, b_odds):
            for key, v in bb.items():
                row[key] = v["est"]
                row[key + "_lo"], row[key + "_hi"] = v["ci"]
                row[key + "_se"] = v["se"]
                samples.setdefault(key, []).append(v["samples"])
        rows.append(row)
        print(f"    {s}: Poisson {row['ll_poisson']:.4f}  base {row['ll_base']:.4f}  PS {row['ll_ps']:.4f} (n={row['n_odds']})"
              f"  Poisson-PS {row['ll_poisson_o-ll_ps']:+.4f} [{row['ll_poisson_o-ll_ps_lo']:+.4f}, {row['ll_poisson_o-ll_ps_hi']:+.4f}]")
    seas = pd.DataFrame(rows)

    def mean10(key):
        """Mean over the 10 test seasons; CI from the per-season match bootstraps (seasons independent)."""
        v = np.mean(np.vstack(samples[key]), axis=0)
        return {"est": float(seas[key].mean()), "ci": pct(v), "se": float(v.std(ddof=1))}

    # Poisson - Pinnacle across seasons
    d_s = seas["ll_poisson_o-ll_ps"].to_numpy()
    se_s = seas["ll_poisson_o-ll_ps_se"].to_numpy()
    n_s = len(d_s)
    sd_between = float(d_s.std(ddof=1))
    sd_ci = [float(np.sqrt((n_s - 1) * sd_between ** 2 / stats.chi2.ppf(0.975, n_s - 1))),
             float(np.sqrt((n_s - 1) * sd_between ** 2 / stats.chi2.ppf(0.025, n_s - 1)))]
    wts = 1 / se_s ** 2
    d_w = float(np.sum(wts * d_s) / wts.sum())
    Q = float(np.sum(wts * (d_s - d_w) ** 2))
    Q_p = float(stats.chi2.sf(Q, n_s - 1))
    t_ci = [float(d_s.mean() - stats.t.ppf(0.975, n_s - 1) * sd_between / np.sqrt(n_s)),
            float(d_s.mean() + stats.t.ppf(0.975, n_s - 1) * sd_between / np.sqrt(n_s))]
    detect_season = Z80 * se_s
    detect_mean = float(Z80 * np.sqrt(np.sum(se_s ** 2)) / n_s)
    n_poisson_better = int(np.sum(d_s < 0))

    # ------------------------------------------------------------ 4. calibration
    print("[4] calibration")
    rel_pois = R.reliability_table(P[["pH", "pD", "pA"]].to_numpy(), P["oi"].to_numpy(), 5)
    rel_pois.insert(0, "forecaster", "poisson")
    has_all = P[["p_PSC_H", "p_PSC_D", "p_PSC_A"]].notna().all(axis=1).to_numpy()
    rel_psc = R.reliability_table(P.loc[has_all, ["p_PSC_H", "p_PSC_D", "p_PSC_A"]].to_numpy(), P.loc[has_all, "oi"].to_numpy(), 5)
    rel_psc.insert(0, "forecaster", "pinnacle_closing")
    rel = pd.concat([rel_pois, rel_psc], ignore_index=True)
    rel["inside_ci"] = (rel["mean_pred"] >= rel["obs_lo"]) & (rel["mean_pred"] <= rel["obs_hi"])
    citl10 = {k: mean10(f"citl_{k}") for k in "HDA"}
    citl10_psc = {k: mean10(f"citl_psc_{k}") for k in "HDA"}
    goals10 = {"home": mean10("res_home_goals"), "away": mean10("res_away_goals")}
    frozen = json.load(open(os.path.join(RESULTS, "numbers.json")))["preliminary"]
    r24 = seas.set_index("season").loc[PARAM_SEASON]
    hg24 = P.loc[P["season"] == PARAM_SEASON, "home_goals"].to_numpy(float)
    rng_g = np.random.default_rng(SEED + 6000)
    hg_boot = hg24[rng_g.integers(0, len(hg24), size=(N_BOOT_EVAL, len(hg24)))].mean(axis=1)
    frozen_goal_ci = sorted(pct(frozen["poisson_mean_lambda"] - hg_boot))

    # ------------------------------------------------------------ 5. parameter uncertainty, 2024-25
    print(f"[5] training-set bootstrap for {PARAM_SEASON}: {N_BOOT_PARAM} replicates x full rolling refits")
    rng_p = np.random.default_rng(SEED + 2000)
    k_p = L.seasons.index(PARAM_SEASON)
    test_idx = np.flatnonzero(L.s == k_p)
    oi_p = O[test_idx]
    ll_fixed, ll_total, lam_mean = np.empty(N_BOOT_PARAM), np.empty(N_BOOT_PARAM), np.empty(N_BOOT_PARAM)
    tb = time.time()
    for b in range(N_BOOT_PARAM):
        m = R.stratified_multiplicities(L.s, rng_p, upto=k_p)
        r = R.rolling_forecast(L, PARAM_SEASON, best_h, best_g, "relegated", mult=m)
        n_fail_total += r["n_fail"]
        lv = R.per_match_logloss(r["p"], oi_p)
        ll_fixed[b] = lv.mean()
        ll_total[b] = np.average(lv, weights=m[test_idx])
        lam_mean[b] = r["lam"].mean()
        if (b + 1) % 50 == 0:
            print(f"    {b + 1}/{N_BOOT_PARAM}  ({time.time() - tb:.0f}s)")
    match_only_ci = [r24["ll_poisson_lo"], r24["ll_poisson_hi"]]
    total_ci = pct(ll_total)
    width_ratio = (total_ci[1] - total_ci[0]) / (match_only_ci[1] - match_only_ci[0])

    # ------------------------------------------------------------ 6. betting
    print("[6] flat-stake betting at Pinnacle closing odds")
    rng_b = np.random.default_rng(SEED + 3000)
    bet_rows = []
    for margin in MARGINS:
        for label, ss in [(s, [s]) for s in BET_SEASONS] + [("2024-26 pooled", BET_SEASONS)]:
            d = P[P["season"].isin(ss)].dropna(subset=["PSCH", "PSCD", "PSCA"])
            prof, nb = R.flat_stake_profit(d[["pH", "pD", "pA"]].to_numpy(), d[["PSCH", "PSCD", "PSCA"]].to_numpy(),
                                           d["oi"].to_numpy(), margin)
            br = R.betting_return(prof, nb, rng_b, N_BOOT_BET)
            bet_rows.append({"margin": margin, "seasons": label, "n_matches": len(d), "n_bets": br["n_bets"],
                             "return_per_unit": br["est"], "lo": br["ci"][0], "hi": br["ci"][1],
                             "profit_units": float(prof.sum())})
    bets = pd.DataFrame(bet_rows)
    print(bets.to_string(index=False))

    # ------------------------------------------------------------ 7. figures
    print("[7] figures")
    F.setup_style()
    R.fig6_rolling_logloss(seas, d_s, best, N_BOOT_EVAL)
    R.fig7_rolling_reliability(rel)

    runtime = time.time() - t0

    # ------------------------------------------------------------ 8. outputs
    lab = {None: "no decay"}
    def how_mean(what):
        return (f"mean over the 10 test seasons of the per-season mean {what}; 95% CI from {N_BOOT_EVAL} "
                "match-resampling bootstrap replicates per season, averaged across seasons replicate by replicate")
    how_ll = how_mean("log-loss (nats/match)")
    how_br = how_mean("Brier score")
    how_citl = how_mean("difference (predicted probability - realised share)")
    how_goals = how_mean("difference (goals per match)")
    pin_note = "Pinnacle and the paired Poisson values use the 3,630 matches with odds (2025-26: 210 of 380)"
    head = {
        "logloss_poisson_mean10": {**mean10("ll_poisson"), "n": int(seas["n"].sum()),
                                   "how": f"rolling Poisson ({best}), refitted before every Fri-Mon / Tue-Thu window; " + how_ll},
        "logloss_base_rate_mean10": {**mean10("ll_base"), "n": int(seas["n"].sum()),
                                     "how": "H/D/A frequencies of all seasons before the test season (from 2005-06); " + how_ll},
        "logloss_pinnacle_prematch_mean10": {**mean10("ll_ps"), "n": int(seas["n_odds"].sum()),
                                             "how": "Pinnacle pre-match (Fri/Tue collection, 1-3 days before kick-off), margin removed proportionally; "
                                                    + pin_note + "; " + how_ll},
        "logloss_pinnacle_closing_mean10": {**mean10("ll_psc"), "n": int(seas["n_odds"].sum()),
                                            "how": "Pinnacle closing odds, margin removed proportionally; " + pin_note + "; " + how_ll},
        "logloss_poisson_on_odds_matches_mean10": {**mean10("ll_poisson_o"), "n": int(seas["n_odds"].sum()),
                                                   "how": "same Poisson forecasts restricted to matches with Pinnacle odds; " + how_ll},
        "brier_poisson_mean10": {**mean10("br_poisson"), "n": int(seas["n"].sum()), "how": "multi-class Brier (0..2); " + how_br},
        "brier_base_rate_mean10": {**mean10("br_base"), "n": int(seas["n"].sum()), "how": "multi-class Brier (0..2); " + how_br},
        "brier_pinnacle_prematch_mean10": {**mean10("br_ps"), "n": int(seas["n_odds"].sum()), "how": "multi-class Brier; " + pin_note + "; " + how_br},
        "poisson_minus_base_logloss_mean10": {**mean10("ll_poisson-ll_base"), "n": int(seas["n"].sum()),
                                              "how": "paired difference (negative = Poisson better), all 3,800 matches; " + how_ll},
        "poisson_minus_pinnacle_prematch_logloss_mean10": {**mean10("ll_poisson_o-ll_ps"), "n": int(seas["n_odds"].sum()),
                                                           "how": "paired difference (positive = Pinnacle better); " + pin_note + "; " + how_ll},
        "poisson_minus_pinnacle_prematch_logloss_mean10_seasons_as_units": {
            "est": float(d_s.mean()), "ci": t_ci, "n": n_s,
            "how": "same mean, CI = t interval over the 10 per-season differences (treats seasons as the sampling unit)"},
        "poisson_minus_pinnacle_closing_logloss_mean10": {**mean10("ll_poisson_o-ll_psc"), "n": int(seas["n_odds"].sum()),
                                                          "how": "paired difference vs Pinnacle closing; " + pin_note + "; " + how_ll},
        "poisson_minus_pinnacle_between_season_sd": {
            "est": sd_between, "ci": sd_ci, "n": n_s,
            "how": f"SD of the 10 per-season Poisson - Pinnacle pre-match differences; chi-square CI; Cochran Q = {Q:.1f} on {n_s - 1} df "
                   f"(p = {Q_p:.2f}) against the per-season bootstrap SEs"},
        "poisson_minus_pinnacle_detectable_gap_per_season": {
            "est": float(detect_season.mean()), "ci": [float(detect_season.min()), float(detect_season.max())], "n": n_s,
            "how": f"2.80 x per-season paired bootstrap SE (5% two-sided test, 80% power); est = mean over seasons, "
                   f"'ci' = range over the 10 seasons (a design quantity, not a confidence interval); mean SE = {se_s.mean():.4f}"},
        "poisson_minus_pinnacle_detectable_gap_10_seasons": {
            "est": detect_mean, "ci": [detect_mean, detect_mean], "n": n_s,
            "how": "2.80 x SE of the 10-season mean difference, SE = sqrt(sum SE_s^2)/10; exact design quantity, no interval"},
        "seasons_poisson_beats_pinnacle_prematch": {
            "est": n_poisson_better, "ci": [float(stats.binom.ppf(0.025, n_s, n_poisson_better / n_s)),
                                            float(stats.binom.ppf(0.975, n_s, n_poisson_better / n_s))] if 0 < n_poisson_better < n_s else [n_poisson_better, n_poisson_better],
            "n": n_s, "how": "count of test seasons with Poisson - Pinnacle pre-match < 0 (point estimate; none is significant on its own unless its CI excludes 0); "
                             "'ci' = binomial 2.5/97.5% quantiles at the observed share"},
        "best_half_life_days": {
            "est": best_h, "ci": hl_ok_range, "n": 3800,
            "how": f"half-life (null = no decay) with the lowest mean log-loss over the 10 test seasons among {len(HALF_LIVES)} x 2 configs; "
                   f"selected on the test seasons themselves. Best on the first five seasons: {best_first5}; on the last five: {best_last5}. "
                   f"'ci' = shortest and longest half-life (days) whose paired log-loss difference from the best has a 95% CI including 0 "
                   f"(same gamma mode; no decay within: {none_ok})"},
        "decay_gain_vs_no_decay": {
            "est": decay_gain["est"], "ci": decay_gain["ci"], "n": 3800,
            "how": f"paired log-loss difference no decay - best half-life (positive = decay helps), all 3,800 test matches; "
                   f"{N_BOOT_EVAL} paired match-bootstrap replicates"},
        "season_gamma_minus_one_gamma": {
            **({"est": -cfg_boot[f"{cfg_name(best_h, 'one')}-{best}"]["est"],
                "ci": sorted(-c for c in cfg_boot[f"{cfg_name(best_h, 'one')}-{best}"]["ci"])} if best_g == "season" else
               {"est": cfg_boot[f"{cfg_name(best_h, 'season')}-{best}"]["est"],
                "ci": cfg_boot[f"{cfg_name(best_h, 'season')}-{best}"]["ci"]}),
            "n": 3800, "how": "paired log-loss difference season-specific gamma - one gamma at the best half-life (negative = season gamma better); "
                              f"{N_BOOT_EVAL} paired match-bootstrap replicates"},
        "promoted_rule_range_all_matches": {
            **{k: rule_range[k] for k in ("est", "ci")}, "n": 3800,
            "how": f"log-loss of the worst rule ({r_worst}) - best rule ({r_best}) over all 3,800 test matches; paired bootstrap, "
                   f"{N_BOOT_EVAL} replicates; rules: mean of relegated / league average / 50% shrink; prior SD {R.DEFAULT_PRIOR_SD} on log scale"},
        "promoted_rule_range_promoted_matches": {
            **rule_range_promo, "n": int(promo_mask.sum()),
            "how": f"same difference on the {int(promo_mask.sum())} test matches involving a promoted team; paired bootstrap"},
        "logloss_2024_25_match_only": {
            "est": float(r24["ll_poisson"]), "ci": [float(c) for c in match_only_ci], "n": 380,
            "how": f"rolling Poisson ({best}); parameters treated as fixed, {N_BOOT_EVAL} resamples of the 380 test matches"},
        "logloss_2024_25_param_and_match": {
            "est": float(r24["ll_poisson"]), "ci": total_ci, "n": 380,
            "how": f"{N_BOOT_PARAM} replicates: resample every season up to 2024-25 with replacement (within season), redo all 45 rolling "
                   "refits incl. the promoted-team targets, score the resampled test matches; percentile CI. Includes parameter, "
                   f"imputation and match sampling uncertainty; width ratio to match-only = {width_ratio:.2f}"},
        "logloss_2024_25_param_only_sd": {
            "est": float(ll_fixed.std(ddof=1)), "ci": pct(ll_fixed - ll_fixed.mean() + float(r24["ll_poisson"])), "n": 380,
            "how": "SD over the training-set bootstrap of the log-loss on the fixed 380 test matches (parameter + imputation only); "
                   "'ci' is the percentile interval of that log-loss re-centred on the point estimate"},
        "citl_home_mean10": {**citl10["H"], "n": 3800, "how": "mean predicted P(home) - realised home share, rolling Poisson; " + how_citl},
        "citl_draw_mean10": {**citl10["D"], "n": 3800, "how": "mean predicted P(draw) - realised draw share; " + how_citl},
        "citl_away_mean10": {**citl10["A"], "n": 3800, "how": "mean predicted P(away) - realised away share; " + how_citl},
        "citl_draw_pinnacle_closing_mean10": {**citl10_psc["D"], "n": int(seas["n_odds"].sum()),
                                              "how": "same for Pinnacle closing, " + pin_note},
        "home_goals_pred_minus_realised_mean10": {**goals10["home"], "n": 3800, "how": "mean lambda - mean home goals; " + how_goals},
        "away_goals_pred_minus_realised_mean10": {**goals10["away"], "n": 3800, "how": "mean mu - mean away goals; " + how_goals},
        "home_goals_pred_minus_realised_2024_25_rolling": {
            "est": float(r24["res_home_goals"]), "ci": [float(r24["res_home_goals_lo"]), float(r24["res_home_goals_hi"])], "n": 380,
            "how": f"rolling mean lambda {r24['mean_lambda']:.3f} vs realised {r24['realised_home_goals']:.3f}; match bootstrap"},
        "home_goals_pred_minus_realised_2024_25_frozen": {
            "est": float(frozen["poisson_mean_lambda"] - frozen["realised_mean_home_goals"]), "ci": frozen_goal_ci, "n": 380,
            "how": f"frozen 2023-24 fit from results/numbers.json 'preliminary': mean lambda {frozen['poisson_mean_lambda']:.3f} vs realised "
                   f"{frozen['realised_mean_home_goals']:.3f}; CI from {N_BOOT_EVAL} resamples of the 380 matches' home goals (lambda fixed)"},
    }
    for _, br in bets.iterrows():
        if br["seasons"] == "2024-26 pooled":
            head[f"betting_return_margin_{int(round(br['margin'] * 100))}pct"] = {
                "est": br["return_per_unit"], "ci": [br["lo"], br["hi"]], "n": int(br["n_bets"]),
                "how": f"flat 1-unit bets at Pinnacle closing odds when p_model * odds - 1 > {br['margin']:.2f}, 2024-25 and 2025-26 "
                       f"({br['n_matches']} matches with odds); return per unit staked; {N_BOOT_BET} match-resampling bootstrap replicates"}

    out = {
        "meta": {
            "seed": SEED, "generated": dt.datetime.now().isoformat(timespec="seconds"), "runtime_seconds": round(runtime, 1),
            "n_boot": {"eval_per_season": N_BOOT_EVAL, "param_training_bootstrap": N_BOOT_PARAM, "betting": N_BOOT_BET},
            "rng_seeds": {"per_season_eval": f"{SEED} + 1000 + index of test season (0..9)", "param_bootstrap": SEED + 2000,
                          "betting": SEED + 3000, "config_comparison": SEED + 4000, "promoted_rules": SEED + 5000,
                          "frozen_goal_ci": SEED + 6000},
            "description": ("Rolling-origin evaluation of the independent-Poisson model, test seasons 2016-17..2025-26. Before every "
                            "Fri-Mon or Tue-Thu matchday window, refit on all earlier matches since 2005-06 with exponential time "
                            "decay; team strengths per spell (consecutive PL seasons); promoted spells get a Gaussian prior "
                            f"(SD {R.DEFAULT_PRIOR_SD} on log scale) toward a rule-based target. Probabilities from the renormalised "
                            "0..10 score grid. Base rate = H/D/A frequencies over all seasons before the test season. Pinnacle "
                            "'pre-match' odds are football-data's Fri/Tue-afternoon collection (1-3 days before kick-off), not opening "
                            "odds; the bookmaker knows injuries and team news the model does not. 2025-26 has Pinnacle odds for 210 of "
                            "380 matches (none after 17 Jan 2026); Pinnacle comparisons use those 210, Poisson vs base rate uses all 380."),
            "primary_config": best, "test_seasons": TEST_SEASONS, "half_lives_days": HALF_LIVES, "gamma_modes": GAMMA_MODES,
            "promoted_rules": list(R.PROMOTED_RULES), "promoted_prior_sd": R.DEFAULT_PRIOR_SD,
            "min_gamma_matches": 100, "min_rel_weight": R.MIN_REL_WEIGHT,
            "optimizer_non_converged_fits_in_param_bootstrap": n_fail_total,
            "optimizer_non_converged_fits_grid": int(sum(v.groupby("season")["n_fail"].first().sum() for v in preds.values())),
        },
        "headline": head,
        "config_grid": {"mean_logloss": grid_mean.to_dict(), "by_season": grid.to_dict(),
                        "paired_vs_best": {k: {"est": v["est"], "ci": v["ci"]} for k, v in cfg_boot.items() if "-" in k and k.endswith(best)},
                        "best_first5": best_first5, "best_last5": best_last5},
        "promoted_rules": {"all_matches": {k: {"est": v["est"], "ci": v["ci"]} for k, v in rule_all.items()},
                           "promoted_matches": {k: {"est": v["est"], "ci": v["ci"]} for k, v in rule_promo.items()},
                           "n_promoted_matches": int(promo_mask.sum())},
        "per_season": seas.to_dict(orient="records"),
        "poisson_minus_pinnacle": {"per_season": dict(zip(TEST_SEASONS, d_s)), "per_season_se": dict(zip(TEST_SEASONS, se_s)),
                                   "precision_weighted_mean": d_w, "cochran_Q": Q, "cochran_Q_df": n_s - 1, "cochran_Q_p": Q_p,
                                   "detectable_gap_per_season": dict(zip(TEST_SEASONS, detect_season))},
        "calibration": {"citl_poisson_mean10": citl10, "citl_pinnacle_closing_mean10": citl10_psc, "goals_mean10": goals10,
                        "reliability": rel.to_dict(orient="records"),
                        "reliability_bins_outside_wilson_ci": {f: int((~rel.loc[rel.forecaster == f, "inside_ci"]).sum())
                                                               for f in ("poisson", "pinnacle_closing")}},
        "param_bootstrap_2024_25": {"ll_fixed_test_set": ll_fixed, "ll_resampled_test_set": ll_total,
                                    "mean_lambda": lam_mean, "width_ratio": width_ratio,
                                    "match_only_ci": match_only_ci, "total_ci": total_ci},
        "betting": bets.to_dict(orient="records"),
        "frozen_2024_25_reference": {k: frozen[k] for k in ("log_loss", "log_loss_ci", "poisson_mean_lambda", "poisson_mean_mu",
                                                            "realised_mean_home_goals", "realised_mean_away_goals",
                                                            "poisson_mean_probs", "realised_frequencies")},
    }
    with open(os.path.join(RESULTS, "rolling.json"), "w") as fh:
        json.dump(jsonable(out), fh, indent=2)
    seas.to_csv(os.path.join(RESULTS, "rolling_by_season.csv"), index=False)
    grid.to_csv(os.path.join(RESULTS, "rolling_config_grid.csv"), index_label="season")
    rel.to_csv(os.path.join(RESULTS, "rolling_reliability.csv"), index=False)
    bets.to_csv(os.path.join(RESULTS, "rolling_betting.csv"), index=False)
    pd.concat(preds.values()).drop(columns=["n_fail"]).to_csv(os.path.join(RESULTS, "rolling_predictions.csv"), index=False)
    pd.DataFrame({"ll_fixed_test_set": ll_fixed, "ll_resampled_test_set": ll_total, "mean_lambda": lam_mean}).to_csv(
        os.path.join(RESULTS, "rolling_param_bootstrap_2024_25.csv"), index=False)
    print(f"done in {runtime:.0f}s")



if __name__ == "__main__":
    main()
