"""Rolling-origin evaluation of the Poisson match model (Agent A, "forecast").

Model fitted before every matchday window of a test season, on all earlier matches:

    X ~ Poisson(lambda),  Y ~ Poisson(mu),  independent
    log lambda = a[e_home] + b[e_away] + g[s],   log mu = a[e_away] + b[e_home]

* ``e`` is a team *spell* (a maximal run of consecutive Premier League seasons), so a
  club that is relegated and comes back starts with fresh parameters; its first-season
  strength comes from a promoted-team rule (below).
* ``g[s]`` is one log home-advantage for every match (``gamma_mode='one'``) or one per
  season (``gamma_mode='season'``).  In season mode the forecast uses the test season's
  own g once at least ``min_gamma_matches`` of its matches are in the training set, and
  the previous season's g before that.
* Match i gets weight w_i = m_i * 0.5 ** (age_i / h), where age_i is the number of days
  between the match and the start of the matchday window being forecast, h is the
  half-life (None = no decay) and m_i is a bootstrap multiplicity (1 by default).
* Estimation maximises  sum_i w_i [x_i log lambda_i - lambda_i + y_i log mu_i - mu_i]
  minus two quadratic penalties: (1) 0.5 * K * (mean of a over the previous season's 20
  spells)^2, which fixes the one non-identified direction a -> a + c, b -> b - c and
  changes no prediction (the anchor set is held fixed through a test season so the
  promoted-team targets below stay on one scale); (2) for every spell promoted into the test season, a Gaussian prior
  0.5 * [(a - a0)^2 + (b - b0)^2] / prior_sd^2 toward the rule's target (a0, b0), so the
  rule decides the forecast before the club has played and fades as its matches arrive.

Promoted-team rules (target computed from the fit made just before the season starts):
  'relegated'   a0, b0 = log of the arithmetic mean of alpha, beta of the relegated spells
                (the rule of the frozen 2023-24 -> 2024-25 forecast in run_all.py);
  'league_avg'  a0, b0 = mean log-strength of the previous season's 20 spells;
  'shrink50'    the midpoint of the two on the log scale.

Matchday windows: Friday-Monday and Tuesday-Thursday.  football-data.co.uk collects the
Pinnacle pre-match odds on Friday and Tuesday afternoons, so the model is refitted at
the same moments the bookmaker's pre-match prices are recorded (the bookmaker also knows
injuries and team news, which this model never sees).

All randomness enters through an ``rng`` (``numpy.random.Generator``) argument.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .evaluate import outcome_to_index
from .model import outcome_probs

LOG_BOUND = 5.0
SCALE_K = 10.0
MIN_REL_WEIGHT = 1e-3          # matches with decay weight below this are dropped from a fit
DEFAULT_PRIOR_SD = 0.3         # promoted-team prior SD on the log scale (~10 matches of data)
PROMOTED_RULES = ("relegated", "league_avg", "shrink50")
_FRIDAY = pd.Timestamp("2000-01-07")   # a Friday, origin for the matchday windows


# --------------------------------------------------------------------------- data layout
def matchday_windows(dates) -> tuple:
    """Assign each date to a Friday-Monday or Tuesday-Thursday window.

    Returns (window_id per date, window start date per date).  The start is the Friday
    or Tuesday that opens the window; a fit for that window may use only matches dated
    strictly before it.
    """
    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    days = (d - _FRIDAY).dt.days.to_numpy()
    week, off = np.divmod(days, 7)                 # off: 0 Fri, 1 Sat, 2 Sun, 3 Mon, 4 Tue, 5 Wed, 6 Thu
    half = (off >= 4).astype(int)
    wid = 2 * week + half
    start = _FRIDAY + pd.to_timedelta(7 * week + 4 * half, unit="D")
    return wid, pd.DatetimeIndex(start)


def spell_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Per (season, team) the spell label 'Team|first season of the consecutive run'."""
    pos = {s: int(str(s)[:4]) for s in df["season"].unique()}     # season start year
    rows = []
    for team in sorted(set(df["home"]) | set(df["away"])):
        ss = sorted({s for s in df.loc[(df["home"] == team) | (df["away"] == team), "season"]}, key=pos.get)
        start = ss[0]
        for k, s in enumerate(ss):
            if k > 0 and pos[s] != pos[ss[k - 1]] + 1:
                start = s
            rows.append((s, team, f"{team}|{start}", start))
    return pd.DataFrame(rows, columns=["season", "team", "spell", "spell_start"])


@dataclass
class Layout:
    """Integer-encoded match history, sorted by date (one object for all refits)."""
    df: pd.DataFrame          # matches sorted by date, with window columns
    spells: list              # spell labels, index = parameter index
    seasons: list             # season labels, index = season index
    h: np.ndarray
    a: np.ndarray
    s: np.ndarray             # season index per match
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray             # days since the first match (float)
    wid: np.ndarray           # matchday window id
    wstart: np.ndarray        # window start, days since the first match
    spell_start: dict         # spell label -> season it started


def make_layout(df: pd.DataFrame) -> Layout:
    df = df.sort_values("date", kind="stable").reset_index(drop=True).copy()
    sp = spell_labels(df)
    key = {(r.season, r.team): r.spell for r in sp.itertuples()}
    spells = sorted(sp["spell"].unique())
    sidx = {e: i for i, e in enumerate(spells)}
    seasons = sorted(df["season"].unique())
    seas_idx = {s: i for i, s in enumerate(seasons)}
    df["home_spell"] = [key[(s, t)] for s, t in zip(df["season"], df["home"])]
    df["away_spell"] = [key[(s, t)] for s, t in zip(df["season"], df["away"])]
    wid, wstart = matchday_windows(df["date"])
    t0 = df["date"].min()
    df["window"] = wid
    df["window_start"] = wstart
    return Layout(df, spells, seasons,
                  df["home_spell"].map(sidx).to_numpy(), df["away_spell"].map(sidx).to_numpy(),
                  df["season"].map(seas_idx).to_numpy(), df["home_goals"].to_numpy(float),
                  df["away_goals"].to_numpy(float), (df["date"] - t0).dt.days.to_numpy(float), wid,
                  (wstart - t0).days.to_numpy(float),
                  dict(zip(sp["spell"], sp["spell_start"])))


def decay_weights(age_days, half_life_days=None) -> np.ndarray:
    """w = 0.5 ** (age / half-life); all ones when ``half_life_days`` is None (no decay)."""
    age = np.asarray(age_days, float)
    if half_life_days is None or not np.isfinite(half_life_days):
        return np.ones_like(age)
    return 0.5 ** (age / float(half_life_days))


# --------------------------------------------------------------------------- estimation
def negloglik(theta, h, a, g, x, y, w, n_e, n_g, anchor, pen_idx, pen_target, prior_sd):
    """Penalised negative weighted log-likelihood (no factorial terms) and gradient."""
    la, lb, lg = theta[:n_e], theta[n_e:2 * n_e], theta[2 * n_e:]
    eh = la[h] + lb[a] + lg[g]
    ea = la[a] + lb[h]
    lam, mu = np.exp(eh), np.exp(ea)
    f = -np.sum(w * (x * eh - lam + y * ea - mu))
    rx, ry = w * (x - lam), w * (y - mu)
    ga = -(np.bincount(h, rx, n_e) + np.bincount(a, ry, n_e))
    gb = -(np.bincount(a, rx, n_e) + np.bincount(h, ry, n_e))
    gg = -np.bincount(g, rx, n_g)
    m = la[anchor].mean()
    f += 0.5 * SCALE_K * m * m
    ga[anchor] += SCALE_K * m / len(anchor)
    if len(pen_idx):
        da = la[pen_idx] - pen_target[:, 0]
        db = lb[pen_idx] - pen_target[:, 1]
        f += 0.5 * (np.sum(da * da) + np.sum(db * db)) / prior_sd ** 2
        ga[pen_idx] += da / prior_sd ** 2
        gb[pen_idx] += db / prior_sd ** 2
    return f, np.concatenate([ga, gb, gg])


def fit_window(L: Layout, idx, w, anchor, gamma_mode="one", x0=None, pen_idx=(), pen_target=None,
               prior_sd=DEFAULT_PRIOR_SD):
    """Fit on matches ``idx`` with weights ``w``; ``anchor`` = spell indices whose mean
    log-attack is pinned to 0.  Returns (theta, converged)."""
    n_e = len(L.spells)
    n_g = len(L.seasons) if gamma_mode == "season" else 1
    g = L.s[idx] if gamma_mode == "season" else np.zeros(len(idx), int)
    h, a = L.h[idx], L.a[idx]
    pen_idx = np.asarray(pen_idx, int)
    anchor = np.asarray(anchor, int)
    pen_target = np.zeros((0, 2)) if pen_target is None else np.asarray(pen_target, float)
    if x0 is None:
        x0 = np.zeros(2 * n_e + n_g)
        x0[n_e:2 * n_e] = np.log(max(np.average(L.y[idx], weights=w), 1e-3))
        x0[2 * n_e:] = np.log(max(np.average(L.x[idx], weights=w), 1e-3) / max(np.average(L.y[idx], weights=w), 1e-3))
    res = minimize(negloglik, x0, jac=True, method="L-BFGS-B",
                   args=(h, a, g, L.x[idx], L.y[idx], w, n_e, n_g, anchor, pen_idx, pen_target, prior_sd),
                   bounds=[(-LOG_BOUND, LOG_BOUND)] * len(x0),
                   options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7})
    return res.x, bool(res.success)


# --------------------------------------------------------------------------- promoted teams
def promoted_targets(theta, L: Layout, season: str, rule: str):
    """Targets (a0, b0) for every spell that starts in ``season``, from a pre-season fit."""
    assert rule in PROMOTED_RULES, rule
    n_e = len(L.spells)
    la, lb = theta[:n_e], theta[n_e:2 * n_e]
    sidx = {e: i for i, e in enumerate(L.spells)}
    k = L.seasons.index(season)
    prev = L.seasons[k - 1]
    cur_sp = set(L.df.loc[L.df["season"] == season, "home_spell"])
    prev_sp = set(L.df.loc[L.df["season"] == prev, "home_spell"])
    promoted = sorted(e for e in cur_sp if L.spell_start[e] == season)
    relegated = sorted(prev_sp - cur_sp)
    prev_i = [sidx[e] for e in prev_sp]
    rel_i = [sidx[e] for e in relegated]
    t_rel = np.array([np.log(np.mean(np.exp(la[rel_i]))), np.log(np.mean(np.exp(lb[rel_i])))])
    t_avg = np.array([la[prev_i].mean(), lb[prev_i].mean()])
    target = {"relegated": t_rel, "league_avg": t_avg, "shrink50": 0.5 * (t_rel + t_avg)}[rule]
    return [sidx[e] for e in promoted], np.tile(target, (len(promoted), 1)), promoted, relegated


# --------------------------------------------------------------------------- rolling forecast
def rolling_forecast(L: Layout, test_season: str, half_life=None, gamma_mode="one", rule="relegated",
                     prior_sd=DEFAULT_PRIOR_SD, mult=None, min_gamma_matches=100) -> dict:
    """Refit before every matchday window of ``test_season`` and forecast that window.

    ``mult`` (optional, per match of ``L``) multiplies the decay weights, e.g. bootstrap
    counts.  Returns per-test-match lambda, mu, P(H, D, A) and diagnostics.
    """
    n_e = len(L.spells)
    k_season = L.seasons.index(test_season)
    test = np.flatnonzero(L.s == k_season)
    mult = np.ones(len(L.x)) if mult is None else np.asarray(mult, float)
    lam = np.empty(len(test))
    mu = np.empty(len(test))
    windows = np.unique(L.wid[test])
    prev = L.seasons[k_season - 1]
    sidx = {e: i for i, e in enumerate(L.spells)}
    anchor = sorted(sidx[e] for e in set(L.df.loc[L.df["season"] == prev, "home_spell"]))
    theta = None
    pen_idx, pen_target, info = [], None, {}
    n_fail = 0
    for j, wnd in enumerate(windows):
        rows = test[L.wid[test] == wnd]
        t_ref = L.wstart[rows[0]]
        train = np.flatnonzero(L.t < t_ref)
        w = decay_weights(t_ref - L.t[train], half_life) * mult[train]
        keep = w > MIN_REL_WEIGHT
        train, w = train[keep], w[keep]
        if j == 0:
            # pre-season fit: no promoted spell has data yet; it sets the promoted targets
            theta, ok = fit_window(L, train, w, anchor, gamma_mode, x0=None)
            pen_idx, pen_target, promoted, relegated = promoted_targets(theta, L, test_season, rule)
            info = {"promoted": promoted, "relegated": relegated, "target": pen_target[0].tolist()}
            theta[pen_idx] = pen_target[:, 0]
            theta[[n_e + i for i in pen_idx]] = pen_target[:, 1]
        else:
            theta, ok = fit_window(L, train, w, anchor, gamma_mode, x0=theta, pen_idx=pen_idx, pen_target=pen_target,
                                   prior_sd=prior_sd)
        n_fail += not ok
        la, lb, lg = theta[:n_e], theta[n_e:2 * n_e], theta[2 * n_e:]
        if gamma_mode == "season":
            n_in = int(np.sum(L.s[train] == k_season))
            g_use = lg[k_season] if n_in >= min_gamma_matches else lg[k_season - 1]
        else:
            g_use = lg[0]
        hh, aa = L.h[rows], L.a[rows]
        pos = np.searchsorted(test, rows)
        lam[pos] = np.exp(la[hh] + lb[aa] + g_use)
        mu[pos] = np.exp(la[aa] + lb[hh])
    p = outcome_probs(lam, mu)
    return {"idx": test, "lam": lam, "mu": mu, "p": p, "n_windows": len(windows), "n_fail": n_fail,
            "promoted_info": info, "gamma_last": float(np.exp(g_use))}


# --------------------------------------------------------------------------- scoring
def per_match_logloss(p, oi) -> np.ndarray:
    p = np.asarray(p, float)
    return -np.log(np.clip(p[np.arange(len(oi)), oi], 1e-12, None))


def per_match_brier(p, oi) -> np.ndarray:
    p = np.asarray(p, float)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(oi)), oi] = 1.0
    return np.sum((p - onehot) ** 2, axis=1)


def base_rate_from_prior(df: pd.DataFrame, test_season: str) -> np.ndarray:
    """H/D/A frequencies over every season in ``df`` strictly before ``test_season``."""
    prior = df[df["season"] < test_season]
    oi = outcome_to_index(prior["outcome"].astype(str))
    return np.bincount(oi, minlength=3) / len(oi)


def paired_bootstrap(losses: dict, rng, n_boot=2000, pairs=(), return_samples=False) -> dict:
    """Percentile 95% CIs for the mean of each per-match loss vector and for paired differences.

    All vectors must be on the same matches; one resample of match indices is shared by
    every forecaster (paired).  ``pairs`` = [(a, b), ...] gives mean(a) - mean(b).
    With ``return_samples`` each entry also carries its bootstrap means under "samples".
    """
    keys = list(losses)
    n = len(losses[keys[0]])
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = {k: np.asarray(losses[k])[idx].mean(axis=1) for k in keys}
    out = {k: {"est": float(np.mean(losses[k])), "ci": _pct(boot[k]), "se": float(boot[k].std(ddof=1))} for k in keys}
    for a, b in pairs:
        d = boot[a] - boot[b]
        out[f"{a}-{b}"] = {"est": float(np.mean(losses[a]) - np.mean(losses[b])), "ci": _pct(d),
                           "se": float(d.std(ddof=1))}
        boot[f"{a}-{b}"] = d
    if return_samples:
        for k in out:
            out[k]["samples"] = boot[k]
    return out


def _pct(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# --------------------------------------------------------------------------- calibration
def calibration_in_the_large(p, oi) -> dict:
    """Mean predicted probability minus realised frequency, per outcome H, D, A."""
    p = np.asarray(p, float)
    return {k: float(p[:, i].mean() - np.mean(oi == i)) for i, k in enumerate("HDA")}


def wilson_ci(k, n, z=1.959964):
    if n == 0:
        return [np.nan, np.nan]
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(c - hw), float(c + hw)]


def reliability_table(p, oi, n_bins=5) -> pd.DataFrame:
    """Equal-count bins of the predicted probability, separately for H, D and A.

    Columns: outcome, bin, n, p_lo, p_hi, mean_pred, obs_freq, obs_lo, obs_hi (Wilson 95%).
    """
    p = np.asarray(p, float)
    rows = []
    for i, k in enumerate("HDA"):
        q = p[:, i]
        order = np.argsort(q, kind="stable")
        for b, part in enumerate(np.array_split(order, n_bins)):
            hit = int(np.sum(oi[part] == i))
            lo, hi = wilson_ci(hit, len(part))
            rows.append((k, b + 1, len(part), float(q[part].min()), float(q[part].max()), float(q[part].mean()),
                         hit / len(part), lo, hi))
    return pd.DataFrame(rows, columns=["outcome", "bin", "n", "p_lo", "p_hi", "mean_pred", "obs_freq", "obs_lo", "obs_hi"])


# --------------------------------------------------------------------------- betting
def flat_stake_profit(p_model, odds, oi, margin=0.0) -> tuple:
    """One-unit bets on every outcome with p_model * odds - 1 > margin.

    Returns (profit per match summed over its bets, number of bets per match).  A bet on
    outcome k returns odds_k - 1 if k happens and -1 otherwise.
    """
    p_model, odds = np.asarray(p_model, float), np.asarray(odds, float)
    bet = p_model * odds - 1.0 > margin
    win = np.zeros_like(bet)
    win[np.arange(len(oi)), oi] = True
    profit = np.where(bet, np.where(win, odds - 1.0, -1.0), 0.0).sum(axis=1)
    return profit, bet.sum(axis=1)


def betting_return(profit, n_bets, rng, n_boot=2000) -> dict:
    """Return per unit staked = sum(profit) / sum(bets), with a match-resampling bootstrap CI."""
    profit, n_bets = np.asarray(profit, float), np.asarray(n_bets, float)
    tot = n_bets.sum()
    est = float(profit.sum() / tot) if tot else float("nan")
    idx = rng.integers(0, len(profit), size=(n_boot, len(profit)))
    pb, nb = profit[idx].sum(axis=1), n_bets[idx].sum(axis=1)
    r = pb[nb > 0] / nb[nb > 0]
    return {"est": est, "ci": _pct(r) if len(r) else [float("nan")] * 2, "n_bets": int(tot)}


# --------------------------------------------------------------------------- bootstrap of the training set
def stratified_multiplicities(season_idx, rng, upto) -> np.ndarray:
    """Bootstrap counts: resample matches with replacement within each season <= ``upto``.

    Seasons after ``upto`` get weight 0 (they are never used).
    """
    s = np.asarray(season_idx)
    m = np.zeros(len(s))
    for k in range(upto + 1):
        rows = np.flatnonzero(s == k)
        m += np.bincount(rng.choice(rows, size=len(rows), replace=True), minlength=len(s))
    return m


# --------------------------------------------------------------------------- figures
def fig6_rolling_logloss(seas: pd.DataFrame, d_s, best: str, n_boot_eval: int = 2000,
                         name: str = "fig6_rolling_logloss"):
    """Per-season held-out log-loss (left) and the paired Poisson − Pinnacle difference (right).

    ``seas`` is the per-season table written to ``results/rolling_by_season.csv``;
    ``d_s`` the per-season Poisson − Pinnacle pre-match differences; ``best`` the
    primary configuration label (``meta.primary_config`` in ``results/rolling.json``).
    """
    import matplotlib.pyplot as plt

    from . import figures as F
    C = F.C
    x = np.arange(len(seas))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    series = [("ll_base", "Base rate (prior seasons' H/D/A)", C["grey"], -0.24),
              ("ll_poisson", "Poisson, rolling refit", C["blue"], -0.08),
              ("ll_ps", "Pinnacle pre-match (Fri/Tue; 2025-26: 210 of 380 matches)", C["aqua"], 0.08),
              ("ll_psc", "Pinnacle closing", C["ink2"], 0.24)]
    for key, label, col, off in series:
        e = seas[key].to_numpy()
        ax.errorbar(x + off, e, yerr=[e - seas[key + "_lo"], seas[key + "_hi"] - e], fmt="o", color=col, ms=6,
                    capsize=3, lw=1.5, label=label)
    ax.axhline(np.log(3), color=C["ink2"], ls=":", lw=1.1)
    ax.text(-0.45, np.log(3) + 0.004, "uniform: ln 3", ha="left", va="bottom", fontsize=11, color=C["ink2"])
    ax.set_xticks(x)
    ax.set_xticklabels(seas["season"], rotation=45, ha="right")
    ax.set_ylabel("Log-loss (nats per match), lower is better")
    ax.set_title("Held-out log-loss by test season")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=11)
    ax = axes[1]
    d_s = np.asarray(d_s, float)
    lo, hi = seas["ll_poisson_o-ll_ps_lo"].to_numpy(), seas["ll_poisson_o-ll_ps_hi"].to_numpy()
    cols = [C["blue"] if v > 0 else C["orange"] for v in d_s]
    for xi, e, l, h, c in zip(x, d_s, lo, hi, cols):
        ax.errorbar(xi, e, yerr=[[e - l], [h - e]], fmt="o", color=c, ms=7, capsize=4, lw=1.8)
    ax.axhline(0, color=C["ink2"], lw=1.2)
    ax.axhline(d_s.mean(), color=C["blue"], ls="--", lw=1.2)
    ax.text(-0.45, d_s.mean() + 0.001, f"mean {d_s.mean():+.3f}", ha="left", va="bottom", fontsize=11, color=C["blue"])
    ax.set_xticks(x)
    ax.set_xticklabels(seas["season"], rotation=45, ha="right")
    ax.set_ylabel("Poisson − Pinnacle pre-match (nats)")
    ax.set_title("Paired difference (positive = Pinnacle better)")
    hl = best.split("|")[0].replace("hl=", "half-life ")
    fig.suptitle(f"Rolling-origin forecasts, 2016-17 to 2025-26 ({hl}); 95% CIs from {n_boot_eval} paired match-bootstrap "
                 "resamples per season", fontsize=14)
    fig.tight_layout()
    return F.save(fig, name)


def fig7_rolling_reliability(rel: pd.DataFrame, name: str = "fig7_rolling_reliability"):
    """Reliability diagram, 5 equal-count bins per outcome, Poisson (rolling) vs Pinnacle closing.

    ``rel`` is the table written to ``results/rolling_reliability.csv``.
    """
    import matplotlib.pyplot as plt

    from . import figures as F
    C = F.C
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=False)
    names = {"H": "Home win", "D": "Draw", "A": "Away win"}
    for ax, k in zip(axes, "HDA"):
        for f, col, mk, lab, dx in [("poisson", C["blue"], "o", "Poisson, rolling", -0.004),
                                     ("pinnacle_closing", C["aqua"], "s", "Pinnacle closing", 0.004)]:
            r = rel[(rel.forecaster == f) & (rel.outcome == k)]
            ax.errorbar(r["mean_pred"] + dx, r["obs_freq"], yerr=[r["obs_freq"] - r["obs_lo"], r["obs_hi"] - r["obs_freq"]],
                        fmt=mk, color=col, ms=7, capsize=3, lw=1.5, label=lab)
        lo = min(rel.loc[rel.outcome == k, ["mean_pred", "obs_lo"]].min()) - 0.03
        hi = max(rel.loc[rel.outcome == k, ["mean_pred", "obs_hi"]].max()) + 0.03
        ax.plot([lo, hi], [lo, hi], color=C["ink2"], ls="--", lw=1.1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(names[k])
        ax.set_xlabel("Mean predicted probability (bin)")
    axes[0].set_ylabel("Observed frequency (Wilson 95% CI)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Reliability, 5 equal-count bins per outcome, test seasons 2016-17 to 2025-26 "
                 "(Poisson 3,800 matches; Pinnacle 3,630 with odds)", fontsize=14)
    fig.tight_layout()
    return F.save(fig, name)
