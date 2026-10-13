"""Home advantage over time: per-season fits, the goals-ratio identity, common-gamma tests,
block/cluster bootstraps, and the CI-coverage and cold-start recovery studies.

Analytic solution used as a verification check.  For one season of a double round
robin with the constraint geometric-mean(alpha) = 1, the score equations give
    d/d log gamma :  sum_m (x_m - lambda_m) = 0
    sum of d/d log alpha_i and d/d log beta_j over all teams :  sum_m (x_m - lambda_m) + (y_m - mu_m) = 0
so sum lambda = sum x and sum mu = sum y.  In a double round robin every ordered pair
(i, j) appears once, so sum_m alpha_h beta_a = sum_m alpha_a beta_h = P and
sum lambda / sum mu = gamma P / P = gamma.  Hence
    gamma_hat = sum(home goals) / sum(away goals)                       exactly,
    extra home goals per match = mean(lambda - lambda/gamma) = mean H - mean A.
The same argument with season-specific strengths and one shared gamma gives the pooled
common-gamma MLE = (sum of home goals) / (sum of away goals) over all pooled seasons.
Bootstrap replicates (unequal match weights) break the balance, so they need refits.

All randomness goes through an explicit ``rng = np.random.default_rng(seed)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from .crowd import projected_gradient_norm
from .model import LOG_BOUNDS, Design, Spec, fit_poisson, negloglik_and_grad, rates
from .simulate import simulate_scores

SEED = 115


# --------------------------------------------------------------------- small helpers
def percentile_ci(samples, level=0.95):
    a = (1 - level) / 2
    return [float(np.percentile(samples, 100 * a)), float(np.percentile(samples, 100 * (1 - a)))]


def goals_ratio(x, y):
    return float(np.sum(x) / np.sum(y))


def analytic_log_gamma_se(x, y):
    """SE of log(sum x / sum y) when sum x and sum y are independent Poisson counts: sqrt(1/H + 1/A)."""
    return float(np.sqrt(1.0 / np.sum(x) + 1.0 / np.sum(y)))


def fit_diagnostics(theta, design: Design, spec: Spec, weights=None) -> dict:
    """max |theta|, distance of the largest |theta| to the +-5 bound, and the projected-gradient norm."""
    _, g = negloglik_and_grad(theta, design, spec, weights)
    m = float(np.abs(theta).max())
    return {"max_abs_theta": m, "distance_to_bound": LOG_BOUNDS[1] - m,
            "proj_grad_norm": projected_gradient_norm(theta, g)}


def pearson_dispersion(fit, design: Design) -> dict:
    """Pearson X^2 / (2n - p) over home and away goals, with the 95% range expected under the Poisson model."""
    lam, mu = rates(fit.log_alpha, fit.log_beta, fit.log_gamma, design.home, design.away)
    x2 = float(np.sum((design.x - lam) ** 2 / lam) + np.sum((design.y - mu) ** 2 / mu))
    df = 2 * design.n_matches - fit.n_params
    return {"pearson_x2": x2, "df": int(df), "dispersion": x2 / df,
            "poisson_95_range": [float(stats.chi2.ppf(0.025, df) / df), float(stats.chi2.ppf(0.975, df) / df)]}


# ------------------------------------------------------------------ bootstrap weights
def iid_weights(m, n_boot, rng):
    """(n_boot, m) multiplicities from resampling m matches with replacement.

    Draws ``rng.choice(arange(m), m)`` per replicate, the same random stream as
    ``src.bootstrap.bootstrap_fit`` without strata, so the same seed gives the same replicates.
    """
    idx = np.arange(m)
    W = np.empty((n_boot, m), dtype=np.uint16)
    for b in range(n_boot):
        W[b] = np.bincount(rng.choice(idx, size=m, replace=True), minlength=m)
    return W


def date_blocks(dates, block_size=10):
    """Block id per match: sort by date (stable, so same-day ties keep table order), cut into consecutive blocks."""
    order = np.argsort(np.asarray(pd.to_datetime(dates)), kind="stable")
    blocks = np.empty(len(order), dtype=int)
    blocks[order] = np.arange(len(order)) // block_size
    return blocks


def cluster_weights(cluster_ids, n_boot, rng):
    """(n_boot, m) multiplicities from resampling whole clusters with replacement (block or team bootstrap)."""
    codes, uniq = pd.factorize(np.asarray(cluster_ids))
    k = len(uniq)
    W = np.empty((n_boot, len(codes)), dtype=np.uint16)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, k, k), minlength=k)
        W[b] = counts[codes]
    return W


def refit_weights(design: Design, fit, W, spec: Spec | None = None) -> dict:
    """Warm-started weighted refits for every row of W; returns gamma, extra home goals and diagnostics."""
    spec = spec or Spec(design, fit.constraint)
    B = len(W)
    gam, ha = np.empty(B), np.empty(B)
    max_theta, max_pg, conv = 0.0, 0.0, 0
    for b in range(B):
        w = W[b].astype(float)
        fb = fit_poisson(design, weights=w, x0=fit.theta, spec=spec)
        gam[b] = fb.gamma
        lam, _ = rates(fb.log_alpha, fb.log_beta, fb.log_gamma, design.home, design.away)
        ha[b] = np.average(lam - lam / fb.gamma, weights=w)
        dg = fit_diagnostics(fb.theta, design, spec, w)
        max_theta, max_pg = max(max_theta, dg["max_abs_theta"]), max(max_pg, dg["proj_grad_norm"])
        conv += fb.converged
    return {"gamma_samples": gam, "ha_goals_samples": ha, "gamma_ci": percentile_ci(gam),
            "ha_goals_ci": percentile_ci(ha), "gamma_se": float(gam.std(ddof=1)),
            "log_gamma_se": float(np.log(gam).std(ddof=1)), "n_boot": B, "n_converged": int(conv),
            "max_abs_theta": max_theta, "max_proj_grad_norm": max_pg}


# ---------------------------------------------------------------- common-gamma tests
def cochran_q(y, se) -> dict:
    """Cochran's Q for homogeneity of estimates y with standard errors se (fixed-effect, inverse variance)."""
    y, se = np.asarray(y, float), np.asarray(se, float)
    w = 1 / se ** 2
    mu = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - mu) ** 2))
    df = len(y) - 1
    return {"Q": q, "df": df, "p": float(stats.chi2.sf(q, df)), "fe_mean": mu, "fe_se": float(np.sqrt(1 / np.sum(w))),
            "I2": max(0.0, (q - df) / q) if q > 0 else 0.0,
            "q_95_range_under_homogeneity": [float(stats.chi2.ppf(0.025, df)), float(stats.chi2.ppf(0.975, df))]}


def dersimonian_laird(y, se) -> dict:
    """DerSimonian-Laird tau^2, random-effects mean and its SE, and a Q-profile 95% CI for tau."""
    y, se = np.asarray(y, float), np.asarray(se, float)
    w = 1 / se ** 2
    cq = cochran_q(y, se)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (cq["Q"] - cq["df"]) / c)
    wr = 1 / (se ** 2 + tau2)
    mu = float(np.sum(wr * y) / np.sum(wr))
    return {"tau2": tau2, "tau": float(np.sqrt(tau2)), "re_mean": mu, "re_se": float(np.sqrt(1 / np.sum(wr))),
            "tau_ci_qprofile": tau_qprofile_ci(y, se)}


def _q_gen(tau2, y, se):
    w = 1 / (se ** 2 + tau2)
    mu = np.sum(w * y) / np.sum(w)
    return float(np.sum(w * (y - mu) ** 2))


def tau_qprofile_ci(y, se, level=0.95):
    """Q-profile CI for tau (Viechtbauer 2007): tau^2 where the generalised Q equals chi2 quantiles."""
    y, se = np.asarray(y, float), np.asarray(se, float)
    df = len(y) - 1
    hi_q, lo_q = stats.chi2.ppf((1 + level) / 2, df), stats.chi2.ppf((1 - level) / 2, df)
    big = 10.0
    lo = 0.0 if _q_gen(0.0, y, se) <= hi_q else brentq(lambda t: _q_gen(t, y, se) - hi_q, 0.0, big)
    hi = 0.0 if _q_gen(0.0, y, se) <= lo_q else brentq(lambda t: _q_gen(t, y, se) - lo_q, 0.0, big)
    return [float(np.sqrt(lo)), float(np.sqrt(hi))]


def season_z(y_s, se_s, y_ref, se_ref) -> dict:
    """z of one season's log gamma against a reference set, under a common gamma and under random effects.

    common:  z = (y_s - mu_FE) / sqrt(se_s^2 + se(mu_FE)^2)
    random:  z = (y_s - mu_RE) / sqrt(se_s^2 + tau^2 + se(mu_RE)^2), tau^2 by DerSimonian-Laird on the reference set
    """
    cq = cochran_q(y_ref, se_ref)
    dl = dersimonian_laird(y_ref, se_ref)
    sd_fe = np.sqrt(se_s ** 2 + cq["fe_se"] ** 2)
    sd_re = np.sqrt(se_s ** 2 + dl["tau2"] + dl["re_se"] ** 2)
    z_fe = (y_s - cq["fe_mean"]) / sd_fe
    z_re = (y_s - dl["re_mean"]) / sd_re
    return {"z_common": float(z_fe), "p_common_two_sided": float(2 * stats.norm.sf(abs(z_fe))),
            "z_random": float(z_re), "p_random_two_sided": float(2 * stats.norm.sf(abs(z_re))),
            "ratio_to_common": float(np.exp(y_s - cq["fe_mean"])),
            "ratio_to_common_ci": [float(np.exp(y_s - cq["fe_mean"] - 1.96 * sd_fe)),
                                   float(np.exp(y_s - cq["fe_mean"] + 1.96 * sd_fe))],
            "ratio_to_re_mean": float(np.exp(y_s - dl["re_mean"])),
            "ratio_to_re_mean_ci": [float(np.exp(y_s - dl["re_mean"] - 1.96 * sd_re)),
                                    float(np.exp(y_s - dl["re_mean"] + 1.96 * sd_re))],
            "ref_fe_gamma": float(np.exp(cq["fe_mean"])), "ref_re_gamma": float(np.exp(dl["re_mean"])),
            "ref_tau": dl["tau"], "n_ref": len(y_ref)}


def trend_meta_regression(y, se, t) -> dict:
    """Inverse-variance weighted least squares of y (log gamma) on t (season start year).

    Returns the slope per season with a 95% CI (fixed-effect SE, and the SE inflated by
    sqrt(Q_E / df) when the residual heterogeneity Q_E exceeds its df), and Q_E with its p-value.
    """
    y, se, t = (np.asarray(v, float) for v in (y, se, t))
    w = 1 / se ** 2
    tc = t - np.average(t, weights=w)
    X = np.column_stack([np.ones_like(tc), tc])
    XtW = X.T * w
    cov = np.linalg.inv(XtW @ X)
    beta = cov @ XtW @ y
    resid = y - X @ beta
    qe = float(np.sum(w * resid ** 2))
    df = len(y) - 2
    se_fe = float(np.sqrt(cov[1, 1]))
    se_adj = se_fe * float(np.sqrt(max(1.0, qe / df)))
    return {"slope_log_gamma_per_season": float(beta[1]), "slope_se_fixed": se_fe, "slope_se_adjusted": se_adj,
            "slope_ci": [float(beta[1] - 1.96 * se_adj), float(beta[1] + 1.96 * se_adj)],
            "slope_p": float(2 * stats.norm.sf(abs(beta[1] / se_adj))),
            "residual_Q": qe, "residual_df": df, "residual_p": float(stats.chi2.sf(qe, df)),
            "gamma_at_weighted_mean_year": float(np.exp(beta[0])), "weighted_mean_year": float(np.average(t, weights=w))}


def trend_z(y_s, se_s, t_s, y_ref, se_ref, t_ref) -> dict:
    """z of one season against the linear trend fitted to the reference seasons.

    z = (y_s - prediction at t_s) / sqrt(se_s^2 + var(prediction)), with var(prediction) from the
    weighted regression, inflated by max(1, residual Q / df) as in ``trend_meta_regression``.
    """
    y, se, t = (np.asarray(v, float) for v in (y_ref, se_ref, t_ref))
    w = 1 / se ** 2
    X = np.column_stack([np.ones_like(t), t - t.mean()])
    XtW = X.T * w
    cov = np.linalg.inv(XtW @ X)
    beta = cov @ XtW @ y
    qe = float(np.sum(w * (y - X @ beta) ** 2))
    infl = max(1.0, qe / (len(y) - 2))
    xs = np.array([1.0, t_s - t.mean()])
    pred = float(xs @ beta)
    sd = float(np.sqrt(se_s ** 2 + infl * xs @ cov @ xs))
    z = (y_s - pred) / sd
    return {"z_trend": float(z), "p_trend_two_sided": float(2 * stats.norm.sf(abs(z))),
            "trend_predicted_gamma": float(np.exp(pred)),
            "ratio_to_trend": float(np.exp(y_s - pred)),
            "ratio_to_trend_ci": [float(np.exp(y_s - pred - 1.96 * sd)), float(np.exp(y_s - pred + 1.96 * sd))]}


def lr_common_gamma(loglik_separate, loglik_common, k) -> dict:
    """Likelihood-ratio test of one shared gamma (k-1 constraints) against one gamma per season."""
    lr = 2 * (float(np.sum(loglik_separate)) - float(loglik_common))
    return {"LR": lr, "df": k - 1, "p": float(stats.chi2.sf(lr, k - 1)),
            "lr_95_range_under_h0": [float(stats.chi2.ppf(0.025, k - 1)), float(stats.chi2.ppf(0.975, k - 1))]}


# ------------------------------------------------------------- simulation studies
def coverage_study(design: Design, truth_fit, n_sims, n_boot, rng, level=0.95) -> dict:
    """Share of bootstrap 95% intervals that contain the true gamma, over simulated seasons.

    Each simulated season: Poisson scores from the planted parameters on the real fixture
    list, an MLE fit (warm start at the truth), and ``n_boot`` iid-bootstrap refits
    (warm start at that fit).  Three intervals are checked: bootstrap percentile,
    bootstrap-SE Wald on log gamma, and the analytic log(H/A) +- 1.96 sqrt(1/H + 1/A).
    """
    spec = Spec(design, truth_fit.constraint)
    g0 = truth_fit.gamma
    rows = []
    for s in range(n_sims):
        x, y = simulate_scores(truth_fit.log_alpha, truth_fit.log_beta, truth_fit.log_gamma,
                               design.home, design.away, rng)
        d = Design(design.teams, design.home, design.away, x, y, design.groups)
        f = fit_poisson(d, x0=truth_fit.theta, spec=spec)
        diag = fit_diagnostics(f.theta, d, spec)
        W = iid_weights(d.n_matches, n_boot, rng)
        bt = refit_weights(d, f, W, spec)
        lo, hi = percentile_ci(bt["gamma_samples"], level)
        se_log = bt["log_gamma_se"]
        se_an = analytic_log_gamma_se(x, y)
        z = stats.norm.ppf((1 + level) / 2)
        rows.append({"sim": s, "gamma_hat": f.gamma, "goals_ratio": goals_ratio(x, y), "boot_lo": lo, "boot_hi": hi,
                     "boot_se_log": se_log, "analytic_se_log": se_an,
                     "covered_percentile": lo <= g0 <= hi,
                     "covered_wald_log": abs(np.log(f.gamma) - np.log(g0)) <= z * se_log,
                     "covered_analytic": abs(np.log(goals_ratio(x, y)) - np.log(g0)) <= z * se_an,
                     "max_abs_theta": max(diag["max_abs_theta"], bt["max_abs_theta"]),
                     "max_proj_grad_norm": max(diag["proj_grad_norm"], bt["max_proj_grad_norm"]),
                     "n_converged": int(f.converged) + bt["n_converged"]})
    tab = pd.DataFrame(rows)
    out = {"true_gamma": g0, "n_sims": n_sims, "n_boot": n_boot, "level": level, "per_sim": tab}
    for k in ("percentile", "wald_log", "analytic"):
        c = int(tab[f"covered_{k}"].sum())
        ci = stats.binomtest(c, n_sims).proportion_ci(0.95, method="exact")
        out[f"coverage_{k}"] = {"share": c / n_sims, "n_covered": c, "ci": [float(ci.low), float(ci.high)]}
    return out


def cold_vs_warm_recovery(design: Design, truth_fit, n_sims, rng) -> dict:
    """Refit each simulated season twice: warm start at the truth and cold start (default x0)."""
    spec = Spec(design, truth_fit.constraint)
    g0 = truth_fit.gamma
    gw, gc, diff_theta, iters_c, iters_w = (np.empty(n_sims) for _ in range(5))
    max_theta, max_pg, conv = 0.0, 0.0, 0
    for s in range(n_sims):
        x, y = simulate_scores(truth_fit.log_alpha, truth_fit.log_beta, truth_fit.log_gamma,
                               design.home, design.away, rng)
        d = Design(design.teams, design.home, design.away, x, y, design.groups)
        fw = fit_poisson(d, x0=truth_fit.theta, spec=spec)
        fc = fit_poisson(d, spec=spec)
        gw[s], gc[s] = fw.gamma, fc.gamma
        diff_theta[s] = np.abs(fw.theta - fc.theta).max()
        iters_w[s], iters_c[s] = fw.n_iter, fc.n_iter
        for f in (fw, fc):
            dg = fit_diagnostics(f.theta, d, spec)
            max_theta, max_pg = max(max_theta, dg["max_abs_theta"]), max(max_pg, dg["proj_grad_norm"])
            conv += f.converged
    se = lambda v: float(v.std(ddof=1) / np.sqrt(n_sims))  # noqa: E731
    return {"true_gamma": g0, "n_sims": n_sims,
            "warm": {"gamma_bias": float(gw.mean() - g0), "mc_se": se(gw), "gamma_sd": float(gw.std(ddof=1)),
                     "bias_ci": [float(gw.mean() - g0 - 1.96 * se(gw)), float(gw.mean() - g0 + 1.96 * se(gw))],
                     "mean_iter": float(iters_w.mean())},
            "cold": {"gamma_bias": float(gc.mean() - g0), "mc_se": se(gc), "gamma_sd": float(gc.std(ddof=1)),
                     "bias_ci": [float(gc.mean() - g0 - 1.96 * se(gc)), float(gc.mean() - g0 + 1.96 * se(gc))],
                     "mean_iter": float(iters_c.mean())},
            "max_abs_gamma_diff_cold_vs_warm": float(np.abs(gw - gc).max()),
            "max_abs_theta_diff_cold_vs_warm": float(diff_theta.max()),
            "max_abs_theta": max_theta, "max_proj_grad_norm": max_pg, "n_converged": int(conv),
            "n_fits": 2 * n_sims}


def stratified_iid_weights(strata, n_boot, rng):
    """(n_boot, m) multiplicities from resampling matches with replacement within each stratum (e.g. season)."""
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in pd.unique(strata)]
    W = np.zeros((n_boot, len(strata)), dtype=np.uint16)
    for b in range(n_boot):
        for idx in groups:
            W[b] += np.bincount(rng.choice(idx, size=len(idx), replace=True), minlength=len(strata)).astype(np.uint16)
    return W


# ---------------------------------------------------------------------------- figures
def fig2_home_advantage_2005_2026(table, common, re_band=None, n_boot=2000, name="fig2_home_advantage_2005_2026"):
    """gamma-hat per season with bootstrap CIs, the common-gamma band (crowd seasons) and, optionally,
    the random-effects 95% prediction band for one season's gamma.

    ``table``: DataFrame indexed by season with gamma, gamma_lo, gamma_hi.
    ``common``: dict(est, lo, hi, label).  ``re_band``: dict(lo, hi, label) or None.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from . import figures as F
    F.setup_style()
    C = F.C
    seasons = list(table.index)
    x = np.arange(len(seasons))
    fig, ax = plt.subplots(figsize=(14, 6.6))
    ymin, ymax = min(table["gamma_lo"].min(), 1.0) - 0.05, table["gamma_hi"].max() + 0.12
    ax.set_ylim(ymin, ymax)
    if "2020-21" in seasons:
        i = seasons.index("2020-21")
        ax.axvspan(i - 0.5, i + 0.5, color=C["orange_light"], zorder=0, lw=0)
    if re_band is not None:
        ax.axhspan(re_band["lo"], re_band["hi"], color=C["grey_light"], alpha=0.35, lw=0, zorder=0)
    ax.axhspan(common["lo"], common["hi"], color=C["blue_light"], alpha=0.7, lw=0, zorder=0.5)
    ax.axhline(common["est"], color=C["blue"], ls="--", lw=1.2, zorder=1)
    ax.axhline(1.0, color=C["ink2"], lw=1.4, zorder=1)
    ax.text(len(seasons) - 0.55, 1.0, "γ = 1: no home advantage", va="bottom", ha="right", fontsize=11, color=C["ink2"])
    for xi, s in zip(x, seasons):
        r = table.loc[s]
        c = C["orange"] if s == "2020-21" else (C["aqua"] if s == "2024-25" else C["blue"])
        ax.errorbar(xi, r["gamma"], yerr=[[r["gamma"] - r["gamma_lo"]], [r["gamma_hi"] - r["gamma"]]], fmt="o",
                    color=c, ms=8, capsize=4, lw=1.8, zorder=3)
    handles = [Line2D([], [], marker="o", color=C["blue"], ms=8, lw=1.8, label="season γ̂ = ΣH/ΣA, 95% CI"),
               Line2D([], [], marker="o", color=C["orange"], ms=8, lw=1.8, label="2020-21 (closed doors)"),
               Line2D([], [], marker="o", color=C["aqua"], ms=8, lw=1.8, label="2024-25 (full stadiums)"),
               Patch(color=C["blue_light"], label=common["label"])]
    if re_band is not None:
        handles.append(Patch(color=C["grey_light"], alpha=0.6, label=re_band["label"]))
    ax.legend(handles=handles, loc="upper right", fontsize=10.5, ncol=2)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons, rotation=45, ha="right")
    ax.set_xlabel("Season")
    ax.set_ylabel("Home-advantage multiplier γ̂")
    ax.set_title(f"Premier League home advantage 2005-06 to 2025-26, Poisson MLE per season (380 matches each)\n"
                 f"95% CIs from {n_boot} bootstrap resamples.  Identity: γ̂ = Σ home goals / Σ away goals; "
                 f"extra home goals = mean H − mean A", fontsize=13)
    return F.save(fig, name)


def fig8_crowd_status(rows, diffs, name="fig8_crowd_gamma"):
    """Left: gamma by crowd status per league with CIs.  Right: closed minus fans-allowed difference.

    ``rows``: list of dict(league, status, est, lo, hi, n).  ``diffs``: list of dict(label, est, lo, hi).
    """
    import matplotlib.pyplot as plt

    from . import figures as F
    F.setup_style()
    C = F.C
    col = {"full": C["blue"], "closed": C["orange"], "limited": C["grey"]}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    leagues = list(dict.fromkeys(r["league"] for r in rows))
    xt, xl = [], []
    xpos = 0.0
    for L in leagues:
        for r in [r for r in rows if r["league"] == L]:
            ax.errorbar(xpos, r["est"], yerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], fmt="o", ms=9,
                        capsize=5, lw=2, color=col[r["status"]])
            xt.append(xpos)
            xl.append(f"{L}\n{r['status']}\n(n={r['n']})")
            xpos += 1
        xpos += 0.6
    ax.axhline(1.0, color=C["ink2"], lw=1.2)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=10)
    ax.set_ylabel("Home-advantage multiplier γ")
    ax.set_title("γ by crowd status, 2019-20 to 2021-22\n(season-specific strengths; flag from match date)",
                 fontsize=13)
    ax = axes[1]
    ys = np.arange(len(diffs))[::-1]
    de = np.array([d["est"] for d in diffs])
    ax.errorbar(de, ys, xerr=[de - np.array([d["lo"] for d in diffs]), np.array([d["hi"] for d in diffs]) - de],
                fmt="o", color=C["orange"], ms=9, capsize=5, lw=2)
    ax.axvline(0, color=C["ink2"], lw=1.2)
    ax.set_yticks(ys)
    ax.set_yticklabels([d["label"] for d in diffs], fontsize=11)
    ax.set_xlabel("γ(closed) − γ(fans allowed)")
    ax.set_title("Closed-doors minus fans-allowed\n(95% CI, season-stratified bootstrap)", fontsize=13)
    fig.tight_layout()
    return F.save(fig, name)
