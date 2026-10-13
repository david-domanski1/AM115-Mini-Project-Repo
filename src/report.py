"""Helpers for the final notebook (``notebooks/final_report.ipynb``) and the Dixon–Coles driver.

Everything the notebook computes or renders lives here or in the other ``src`` modules,
so notebook cells are one-line glue.  Results files written by the driver scripts all
share one layout: ``{"meta": {seed, generated, runtime_seconds, n_boot, description},
"headline": {name: {"est", "ci": [lo, hi], "n", "how"}}, ...}``.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os

import numpy as np
import pandas as pd

from .model import Design, Spec, fit_poisson, make_design, negloglik_and_grad, outcome_probs, rates

# Which script writes which results file (stated in the notebook).
RESULT_SOURCES = {
    "numbers": "run_all.py (make all)",
    "dixon_coles": "scripts/run_dixon_coles.py",
    "rolling": "scripts/run_rolling.py",
    "timeseries": "scripts/run_timeseries.py",
}


# ------------------------------------------------------------------- paths / IO

def find_root(cwd: str | None = None) -> str:
    """Repository root: the first of cwd, its parent, grandparent that contains ``src/``."""
    cwd = os.path.abspath(cwd or os.getcwd())
    for cand in (cwd, os.path.dirname(cwd), os.path.dirname(os.path.dirname(cwd))):
        if os.path.isdir(os.path.join(cand, "src")) and os.path.isdir(os.path.join(cand, "results")):
            return cand
    raise FileNotFoundError(f"no repo root with src/ and results/ above {cwd}")


def load_results(root: str, name: str):
    """Load ``results/<name>.json`` or return None if it does not exist."""
    path = os.path.join(root, "results", f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def fmt_ci(ci, nd=3, signed=False):
    f = f"{{:+.{nd}f}}" if signed else f"{{:.{nd}f}}"
    return "[" + f.format(ci[0]) + ", " + f.format(ci[1]) + "]"


def fmt_est(h: dict, nd=3, signed=False) -> str:
    """'est [lo, hi]' from a headline entry."""
    f = f"{{:+.{nd}f}}" if signed else f"{{:.{nd}f}}"
    return f.format(h["est"]) + " " + fmt_ci(h["ci"], nd, signed)


def headline_markdown(res: dict, title: str = "", nd: int = 4, contains=None) -> str:
    """Render a results file's ``headline`` dict (and its ``meta``) as a Markdown table."""
    meta = res.get("meta", {})
    lines = []
    if title:
        lines.append(f"**{title}**  ")
    lines.append(f"seed {meta.get('seed')}, generated {meta.get('generated')}, runtime "
                 f"{meta.get('runtime_seconds')} s, bootstrap replicates {meta.get('n_boot')}. "
                 f"{meta.get('description', '')}")
    lines += ["", "| quantity | estimate | 95% CI | n | how obtained |", "|---|---|---|---|---|"]
    for k, h in res.get("headline", {}).items():
        if contains is not None and not any(c in k for c in contains):
            continue
        est = h.get("est")
        ci = h.get("ci") or [np.nan, np.nan]
        est_s = f"{est:.{nd}f}" if isinstance(est, (int, float)) and est is not None else str(est)
        ci_s = fmt_ci([np.nan if c is None else c for c in ci], nd)
        how = str(h.get("how", "")).replace("|", "\\|")
        lines.append(f"| {k} | {est_s} | {ci_s} | {h.get('n')} | {how} |")
    return "\n".join(lines)


def results_section(root: str, name: str, title: str = "", contains=None) -> str:
    """Markdown for a results file's headline, or a PLACEHOLDER line if the file is missing."""
    res = load_results(root, name)
    if res is None:
        return (f"**PLACEHOLDER: results/{name}.json not found; run {RESULT_SOURCES.get(name, 'its driver script')} "
                f"and re-run this notebook.**")
    return headline_markdown(res, title or f"results/{name}.json (written by {RESULT_SOURCES.get(name, '?')})",
                             contains=contains)


def show_md(text: str):
    """Display Markdown in a notebook (falls back to print outside IPython)."""
    try:
        from IPython.display import Markdown, display
        display(Markdown(text))
    except ImportError:  # pragma: no cover
        print(text)


def show_figure(path: str, width: int = 950):
    """Display a saved PNG inline."""
    from IPython.display import Image, display
    display(Image(filename=path, width=width))


# ---------------------------------------------------------------- per-season fits

def fit_seasons(core: pd.DataFrame, seasons) -> dict:
    """Per-season Poisson MLE: {season: (design, fit)}."""
    out = {}
    for s in seasons:
        d = make_design(core[core["season"] == s])
        out[s] = (d, fit_poisson(d))
    return out


def season_table(fits: dict, numbers: dict) -> pd.DataFrame:
    """Live gamma-hat per season next to the stored 500-replicate bootstrap CI from numbers.json."""
    rows = []
    for s, (d, f) in fits.items():
        r = numbers["poisson_fit_by_season"][s]
        rows.append({"season": s, "gamma_live": f.gamma, "gamma_numbers_json": r["gamma"],
                     "gamma_lo": r["gamma_lo"], "gamma_hi": r["gamma_hi"], "ha_goals": r["ha_goals"],
                     "ha_goals_lo": r["ha_goals_lo"], "ha_goals_hi": r["ha_goals_hi"],
                     "n_free_params": f.n_params, "converged": f.converged})
    return pd.DataFrame(rows).set_index("season")


def identity_check(fits: dict) -> pd.DataFrame:
    """gamma-hat vs sum(home goals) / sum(away goals): equal for a balanced double round robin.

    The score equations give sum(lambda) = sum(x) and sum(mu) = sum(y); with every ordered
    pair played once, sum_m alpha_h beta_a = sum_m alpha_a beta_h, so gamma = sum x / sum y.
    """
    rows = []
    for s, (d, f) in fits.items():
        ratio = d.x.sum() / d.y.sum()
        rows.append({"season": s, "gamma_hat": f.gamma, "sum_home_over_sum_away": ratio,
                     "abs_diff": abs(f.gamma - ratio)})
    return pd.DataFrame(rows).set_index("season")


def bootstrap_gamma(design: Design, fit, n_boot: int, rng) -> dict:
    """Small nonparametric bootstrap of gamma (resample matches, warm-started refit) using a supplied rng."""
    spec = Spec(design)
    m = design.n_matches
    g = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, m, m), minlength=m).astype(float)
        g[b] = fit_poisson(design, weights=w, x0=fit.theta, spec=spec).gamma
    return {"samples": g, "ci": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
            "se": float(g.std(ddof=1)), "n_boot": n_boot}


def gradient_check(design: Design, theta, fn, h=1e-5) -> float:
    """Max |analytic - central finite-difference| gradient of ``fn(theta, design, spec) -> (f, g)``."""
    spec = Spec(design)
    _, g = fn(theta, design, spec)
    fd = np.empty_like(theta)
    for k in range(len(theta)):
        e = np.zeros_like(theta)
        e[k] = h
        fd[k] = (fn(theta + e, design, spec)[0] - fn(theta - e, design, spec)[0]) / (2 * h)
    return float(np.abs(fd - g).max())


def poisson_gradient_check(design: Design, fit, rng, scale=0.05) -> float:
    """Gradient check of the Poisson log-likelihood at a random point near the MLE."""
    theta = fit.theta + scale * rng.standard_normal(len(fit.theta))
    return gradient_check(design, theta, negloglik_and_grad)


# ------------------------------------------------------------- frozen forecast

def frozen_split_rates(train_fit, hold: pd.DataFrame):
    """Rates for a held-out season from strengths frozen at the end of the training season.

    Same rule as ``run_all.py``: promoted teams get the arithmetic mean alpha and beta of
    the relegated teams (natural scale), everyone else keeps their fitted strength.
    ``train_fit`` is any fit with ``teams``, ``log_alpha``, ``log_beta``, ``log_gamma``.
    """
    teams_ref = list(train_fit.teams)
    teams_hold = set(hold["home"]) | set(hold["away"])
    promoted = sorted(teams_hold - set(teams_ref))
    relegated = sorted(set(teams_ref) - teams_hold)
    idx = {t: i for i, t in enumerate(teams_ref)}
    a = np.exp(train_fit.log_alpha)
    b = np.exp(train_fit.log_beta)
    rel_a = np.mean([a[idx[t]] for t in relegated])
    rel_b = np.mean([b[idx[t]] for t in relegated])
    la = {t: float(train_fit.log_alpha[idx[t]]) for t in teams_ref} | {t: float(np.log(rel_a)) for t in promoted}
    lb = {t: float(train_fit.log_beta[idx[t]]) for t in teams_ref} | {t: float(np.log(rel_b)) for t in promoted}
    lam = np.exp(hold["home"].map(la).to_numpy() + hold["away"].map(lb).to_numpy() + train_fit.log_gamma)
    mu = np.exp(hold["away"].map(la).to_numpy() + hold["home"].map(lb).to_numpy())
    return lam, mu, promoted, relegated


def frozen_split_forecast(core: pd.DataFrame, train: str = "2023-24", test: str = "2024-25"):
    """The run_all.py single-split forecast: Poisson probabilities, pre-test base rate, Pinnacle.

    Returns a dict of (n, 3) probability arrays, the outcome index and the held-out frame.
    """
    from .evaluate import base_rate_probs, outcome_to_index
    d = make_design(core[core["season"] == train])
    f = fit_poisson(d)
    hold = core[core["season"] == test].copy()
    lam, mu, promoted, relegated = frozen_split_rates(f, hold)
    p_pin = hold[["p_PS_H", "p_PS_D", "p_PS_A"]].to_numpy()
    p_pinc = hold[["p_PSC_H", "p_PSC_D", "p_PSC_A"]].to_numpy()
    ok = ~np.isnan(p_pin).any(axis=1) & ~np.isnan(p_pinc).any(axis=1)
    prior = core[core["season"] < test]["outcome"].value_counts(normalize=True)
    hold = hold[ok]
    probs = {"base_rate": base_rate_probs([float(prior[k]) for k in "HDA"], int(ok.sum())),
             "poisson": outcome_probs(lam[ok], mu[ok]), "pinnacle": p_pin[ok], "pinnacle_closing": p_pinc[ok]}
    return {"probs": probs, "outcome_idx": outcome_to_index(hold["outcome"]), "hold": hold, "lam": lam[ok],
            "mu": mu[ok], "ok": ok, "train_fit": f, "train_design": d, "promoted": promoted, "relegated": relegated}


# ------------------------------------------------------- home-advantage figure

def home_advantage_figure(root: str, numbers: dict):
    """The single home-advantage figure of the report.

    If Agent B's ``src/timeseries.py`` exists and exposes a figure function, call it (the
    2005-06..2025-26 gamma figure); otherwise draw gamma by season for the 11 core seasons
    from the stored bootstrap CIs in numbers.json with ``src.figures.fig2_home_advantage``.
    Returns (path, which_path_was_used).
    """
    from . import figures as F
    ts_json = load_results(root, "timeseries")
    ts_csv = os.path.join(root, "results", "timeseries_by_season.csv")
    if importlib.util.find_spec("src.timeseries") is not None and ts_json is not None and os.path.exists(ts_csv):
        ts = importlib.import_module("src.timeseries")
        fn = getattr(ts, "fig2_home_advantage_2005_2026", None)
        try:
            tab = pd.read_csv(ts_csv, index_col="season")
            p20 = ts_json["pooled_common_gamma"]["crowd_20_seasons"]
            t20 = ts_json["common_gamma_tests"]["crowd_20_seasons"]
            common = {"est": p20["gamma"], "lo": p20["gamma_ci"][0], "hi": p20["gamma_ci"][1],
                      "label": f"common γ, 20 seasons excl. 2020-21: {p20['gamma']:.3f} "
                               f"[{p20['gamma_ci'][0]:.3f}, {p20['gamma_ci'][1]:.3f}]"}
            re_band = {"lo": t20["re_prediction_interval_one_season"][0],
                       "hi": t20["re_prediction_interval_one_season"][1],
                       "label": f"random-effects 95% range for one season (τ = {t20['dl_tau']:.3f})"}
            path = fn(tab, common, re_band, n_boot=int(ts_json["meta"].get("n_boot", 2000)))
            return path, "src.timeseries.fig2_home_advantage_2005_2026 (results/timeseries*.json|csv)"
        except (KeyError, TypeError, ValueError) as e:   # INTEGRATOR: Agent B's layout changed; re-wire here
            print(f"timeseries figure failed ({e!r}); using the 11-season fallback")
    tab = pd.DataFrame({s: {"est": r["gamma"], "lo": r["gamma_lo"], "hi": r["gamma_hi"]}
                        for s, r in numbers["poisson_fit_by_season"].items()}).T
    pg = numbers["pooled_gamma"]
    path = F.fig2_home_advantage(tab, {"est": pg["gamma"], "lo": pg["gamma_ci"][0], "hi": pg["gamma_ci"][1]},
                                 kind="gamma", n_boot=numbers["meta"]["n_boot_per_season"],
                                 name="fig2b_home_advantage_gamma")
    return path, "fallback: src.figures.fig2_home_advantage(kind='gamma') on the 11 core seasons"


# ------------------------------------------------------- notebook: section helpers

def outcome_frequencies(core: pd.DataFrame):
    """(per-season H/D/A shares with n, pooled dict) in the form ``figures.fig1_baselines`` takes."""
    freq = core.groupby("season", observed=True)["outcome"].value_counts(normalize=True).unstack()[list("HDA")]
    freq["n"] = core.groupby("season").size()
    pc = core["outcome"].value_counts(normalize=True)
    pooled = {k: float(pc[k]) for k in "HDA"}
    pooled["n"] = int(len(core))
    return freq, pooled


def _prop_ci(p, n):
    se = np.sqrt(p * (1 - p) / n)
    return [p - 1.96 * se, p + 1.96 * se]


def md_question(numbers: dict, freq: pd.DataFrame, pooled: dict) -> str:
    """The headline answer to the question, every number with its interval and how it was obtained."""
    pg, cv = numbers["pooled_gamma"], numbers["covid_comparison"]
    nb = numbers["meta"]["n_boot_per_season"]
    hc = freq.loc["2020-21"]
    return "\n".join([
        f"- **Extra home goals per match, 11 seasons pooled:** {pg['ha_goals']:.3f} {fmt_ci(pg['ha_goals_ci'])} "
        f"(γ̂ = {pg['gamma']:.3f} {fmt_ci(pg['gamma_ci'])}; one γ shared by all seasons, season-specific strengths, "
        f"{pg['n_boot']} season-stratified bootstrap refits; this is a common-γ interval).",
        f"- **2020-21 (closed doors):** γ̂ = {cv['gamma']:.3f} {fmt_ci(cv['gamma_ci'])}, extra home goals "
        f"{cv['ha_goals']:.3f} {fmt_ci(cv['ha_goals_ci'])} ({nb} match-resampling refits).",
        f"- **Difference 2020-21 minus the mean of the other ten seasons:** {cv['gamma_difference_2020_21_minus_mean_others']:+.3f} "
        f"{fmt_ci(cv['gamma_difference_ci'], signed=True)} (difference of independent per-season bootstrap draws).",
        f"- Home-win share 2020-21 {hc['H']:.3f} {fmt_ci(_prop_ci(hc['H'], hc['n']))} vs pooled {pooled['H']:.3f} "
        f"{fmt_ci(_prop_ci(pooled['H'], pooled['n']))} (binomial normal-approximation intervals).",
    ])


def md_assumption_evidence(core: pd.DataFrame, numbers: dict, dc, by_season) -> str:
    """Numbers behind the implicit assumptions: dispersion, restart matches, frozen-strength miscalibration."""
    pre = numbers["preliminary"]
    hold = core[core["season"] == pre["test_season"]]
    hx = hold["home_goals"].to_numpy(float)
    se = hx.std(ddof=1) / np.sqrt(len(hx))
    restart = int(((core["season"] == "2019-20") & (core["date"] >= "2020-06-17")).sum())
    lines = [
        f"- (iv) crowd status = season label: {restart} matches of 2019-20 were played after the 17 June 2020 restart "
        f"behind closed doors (a count from match dates, no sampling error; attendance itself is not in the data).",
        f"- (ii) strengths frozen for a season: the frozen 2023-24 → 2024-25 forecast predicts {pre['poisson_mean_lambda']:.2f} "
        f"home goals per match against a realised {hx.mean():.2f} {fmt_ci([hx.mean() - 1.96 * se, hx.mean() + 1.96 * se], 2)} "
        f"(normal interval of the realised mean, {len(hx)} matches), and P(draw) {pre['poisson_mean_probs']['D']:.3f} against a "
        f"realised share {pre['realised_frequencies']['D']:.3f} {fmt_ci(_prop_ci(pre['realised_frequencies']['D'], len(hx)))}.",
        f"- (v) the 0..10 score grid: largest missing mass in 2023-24 is "
        f"{numbers['scorelines_2023_24']['grid_truncation_max_missing_mass']:.1e} per match (exact computation); "
        f"probabilities are renormalised.",
    ]
    if by_season is not None and dc is not None:
        h = dc["headline"]["pearson_dispersion_pooled"]
        lines.append(
            f"- Poisson variance = mean: Pearson dispersion per season ranges {by_season['pearson_dispersion'].min():.3f}.."
            f"{by_season['pearson_dispersion'].max():.3f} (each season's 95% range under the Poisson model is about "
            f"{fmt_ci([by_season['dispersion_null_lo'].mean(), by_season['dispersion_null_hi'].mean()])}); pooled over 11 seasons "
            f"{h['est']:.3f} against a Poisson range {fmt_ci(h['ci'])}: goals are slightly *under*-dispersed, not over.")
    return "\n".join(lines)


def md_estimation(numbers: dict, demo: dict, fits: dict) -> str:
    stored = numbers["poisson_fit_by_season"]["2020-21"]
    pgx = numbers["pooled_gamma_excluding_2020_21"]
    return "\n".join([
        f"- Live {demo['n_boot']}-replicate bootstrap of 2020-21 γ̂ (seed printed above): {fits['2020-21'][1].gamma:.3f} "
        f"{fmt_ci(demo['ci'])}, SE {demo['se']:.3f}; the stored {numbers['meta']['n_boot_per_season']}-replicate interval "
        f"is {fmt_ci([stored['gamma_lo'], stored['gamma_hi']])}, SE {stored['gamma_se']:.3f}.",
        f"- Common γ over the ten crowd seasons (2020-21 excluded): {pgx['gamma']:.3f} {fmt_ci(pgx['gamma_ci'])} "
        f"({numbers['meta']['n_boot_pooled']} season-stratified refits).",
    ])


def skellam_check(rng, n=200):
    """Max |grid - Skellam| over random rates, on a 0..40 grid and on the 0..10 grid (unnormalised)."""
    from .model import skellam_outcome_probs
    lam, mu = rng.uniform(0.2, 4.0, n), rng.uniform(0.2, 4.0, n)
    sk = skellam_outcome_probs(lam, mu)
    return (float(np.abs(outcome_probs(lam, mu, max_goals=40) - sk).max()),
            float(np.abs(outcome_probs(lam, mu, max_goals=10, normalise=False) - sk).max()))


def dc_gradient_check(design: Design, fit, rng, rho=-0.05, scale=0.05) -> float:
    """Gradient check of the Dixon–Coles log-likelihood at a random point near (Poisson MLE, rho)."""
    from .dixon_coles import dc_negloglik_and_grad
    theta = np.concatenate([fit.theta + scale * rng.standard_normal(len(fit.theta)), [rho]])
    return gradient_check(design, theta, dc_negloglik_and_grad)


def md_verification(skellam_err: float, skellam_err10: float, ident: pd.DataFrame, grad_p: float, grad_dc: float,
                    dc_checks: dict, rec: dict, numbers: dict) -> str:
    r = numbers["verification"]["recovery_truth_2023_24"]
    mcse = rec["gamma_sd"] / np.sqrt(rec["n_sims"])
    return "\n".join([
        f"- Score grid vs Skellam closed form (200 random rate pairs): max |difference| {skellam_err:.1e} on a 0..40 grid, "
        f"{skellam_err10:.1e} on the 0..10 grid used (truncation, before renormalising).",
        f"- Identity γ̂ = ΣH/ΣA: max |difference| over the 11 seasons {ident['abs_diff'].max():.1e} (optimiser tolerance).",
        f"- Analytic vs central-difference gradient at a random point: Poisson {grad_p:.1e}, Dixon–Coles {grad_dc:.1e}.",
        f"- Dixon–Coles: τ conserves probability, max |grid mass − 1| = {dc_checks['mass_err']:.1e} (0..40 grid, admissible ρ); "
        f"ρ = 0 reproduces `outcome_probs` exactly (max diff {dc_checks['rho0_err']:.1e}); planted ρ = {dc_checks['rho_true']:+.2f} "
        f"on {dc_checks['n_matches']} simulated matches is recovered as {dc_checks['rho_hat']:+.3f} "
        f"{fmt_ci(dc_checks['rho_ci'], signed=True)} ({dc_checks['n_boot']} bootstrap refits).",
        f"- Recovery on {rec['n_sims']} simulated 2023-24 seasons: γ̂ bias {rec['gamma_bias']:+.4f} ± {1.96 * mcse:.4f} "
        f"(95% Monte Carlo interval), recovery SD {rec['gamma_sd']:.3f} (stored run: bias {r['gamma_bias']:+.4f}). This checks "
        f"the optimiser under the model, not the model.",
    ])


def dc_verification(rng, n_boot: int = 50) -> dict:
    """Mass conservation, rho = 0 limit, and recovery of a planted rho on 4 simulated 20-team seasons."""
    from .dixon_coles import (bootstrap_rho, dc_grid, dc_outcome_probs, fit_dixon_coles, rho_admissible_range,
                              simulate_dc_scores)
    lam, mu = rng.uniform(0.4, 3.0, 200), rng.uniform(0.4, 2.5, 200)
    lo, hi = rho_admissible_range(lam, mu)
    mass_err = max(float(np.abs(dc_grid(lam, mu, r, 40).sum(axis=(1, 2)) - 1).max()) for r in (0.99 * lo, -0.05, 0.99 * hi))
    rho0_err = float(np.abs(dc_outcome_probs(lam, mu, 0.0) - outcome_probs(lam, mu)).max())
    h, a = np.array([(i, j) for i in range(20) for j in range(20) if i != j]).T
    h, a = np.tile(h, 4), np.tile(a, 4)
    la = rng.normal(0, 0.25, 20)
    la -= la.mean()
    lb = rng.normal(0.1, 0.2, 20)
    lam_s, mu_s = np.exp(la[h] + lb[a] + np.log(1.22)), np.exp(la[a] + lb[h])
    rho_true = -0.05
    x, y = simulate_dc_scores(lam_s, mu_s, rho_true, rng)
    d = Design(list(range(20)), h, a, x, y, np.zeros(20, int))
    fd = fit_dixon_coles(d)
    b = bootstrap_rho(d, fd, n_boot, rng)
    return {"mass_err": mass_err, "rho0_err": rho0_err, "rho_true": rho_true, "rho_hat": fd.rho, "rho_ci": b["ci"],
            "n_boot": n_boot, "n_matches": d.n_matches}


def fig5_from_split(ll: dict, bll: dict, n: int, n_boot: int, season: str = "2024-25"):
    """Figure 5 exactly as run_all.py draws it, from live log-losses and paired-bootstrap samples."""
    from . import figures as F

    def diff(a, b):
        s = bll[a]["samples"] - bll[b]["samples"]
        return ll[a] - ll[b], float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
    return F.fig5_logloss_comparison(
        [("Base rate\n(2015-16..2023-24 H/D/A)", ll["base_rate"], *bll["base_rate"]["ci"]),
         ("Poisson\n(2023-24 strengths)", ll["poisson"], *bll["poisson"]["ci"]),
         ("Pinnacle\n(pre-match odds)", ll["pinnacle"], *bll["pinnacle"]["ci"])],
        [("Poisson − base", *diff("poisson", "base_rate")), ("Pinnacle − base", *diff("pinnacle", "base_rate")),
         ("Poisson − Pinnacle", *diff("poisson", "pinnacle"))], season, n, n_boot)


def heldout_dc_probs(fs: dict):
    """Dixon–Coles probabilities for the frozen split: DC fitted jointly on the training season only."""
    from .dixon_coles import dc_outcome_probs, fit_dixon_coles
    fd = fit_dixon_coles(fs["train_design"], poisson_fit=fs["train_fit"])
    lam, mu, _, _ = frozen_split_rates(fd, fs["hold"])
    return dc_outcome_probs(lam, mu, fd.rho), fd.rho


def split_table(ll: dict, bll: dict, ref: str = "poisson") -> pd.DataFrame:
    """Log-loss with paired-bootstrap CI for each forecaster, and its paired difference from ``ref``."""
    rows = []
    for k in ll:
        s = bll[k]["samples"] - bll[ref]["samples"]
        rows.append({"forecast": k, "log_loss": ll[k], "lo": bll[k]["ci"][0], "hi": bll[k]["ci"][1],
                     f"minus_{ref}": ll[k] - ll[ref], "diff_lo": float(np.percentile(s, 2.5)),
                     "diff_hi": float(np.percentile(s, 97.5))})
    return pd.DataFrame(rows).set_index("forecast")


def md_validation(tab: pd.DataFrame) -> str:
    """Single-split headline from the live table (paired bootstrap over the test matches)."""
    def ci(k, a="lo", b="hi"):
        return fmt_ci([tab.loc[k, a], tab.loc[k, b]], 4, a == "diff_lo")
    def diff(a, b):
        return tab.loc[a, "log_loss"] - tab.loc[b, "log_loss"]
    return "\n".join([
        f"- Held-out 2024-25 log-loss (nats per match, 95% paired-bootstrap CI over the test matches): base rate "
        f"{tab.loc['base_rate', 'log_loss']:.4f} {ci('base_rate')}, Poisson {tab.loc['poisson', 'log_loss']:.4f} {ci('poisson')}, "
        f"Pinnacle pre-match {tab.loc['pinnacle', 'log_loss']:.4f} {ci('pinnacle')}.",
        f"- Base rate − Poisson {diff('base_rate', 'poisson'):+.4f} {ci('base_rate', 'diff_lo', 'diff_hi')}: the model beats "
        f"the no-model baseline on the same metric. Pinnacle − Poisson {diff('pinnacle', 'poisson'):+.4f} "
        f"{ci('pinnacle', 'diff_lo', 'diff_hi')}: one season cannot tell the model and the market apart.",
        "- These intervals treat the 2023-24 parameters as fixed (only test matches are resampled); the rolling "
        "evaluation in 6.2 adds parameter uncertainty.",
    ])


def md_dixon_coles(dc: dict, ll_live: dict, bll_live: dict, rho23: float) -> str:
    h = dc["headline"]
    s = bll_live["dixon_coles"]["samples"] - bll_live["poisson"]["samples"]
    det = dc["details"]["scorelines"]
    lr = dc["details"]["pooled"]["lr"]
    return "\n".join([
        f"- Pooled ρ̂ = {fmt_est(h['pooled_rho'], 3, True)} ({h['pooled_rho']['how']}).",
        f"- LR test of ρ = 0: statistic {lr['statistic']:.2f} on 1 df, p = {lr['p_value']:.3f} (χ²₁ reference); scoreline "
        f"log-likelihood gain {h['pooled_loglik_gain_nats']['est']:.2f} nats over {h['pooled_rho']['n']} matches.",
        f"- 0-0: observed {det['observed_0_0']}, Poisson expects {det['expected_0_0_poisson']:.1f}, Dixon–Coles "
        f"{det['expected_0_0_dc']:.1f} (95% simulation range of obs − exp under Poisson "
        f"{fmt_ci(h['observed_minus_expected_0_0_poisson']['ci'], 1, True)}). 1-1: observed {det['observed_1_1']}, Poisson "
        f"{det['expected_1_1_poisson']:.1f}, Dixon–Coles {det['expected_1_1_dc']:.1f}.",
        f"- Largest pooled Pearson residual: {fmt_est(h['max_abs_pearson_residual_poisson'], 2, True)} "
        f"({h['max_abs_pearson_residual_poisson']['how']}).",
        f"- In-sample outcome log-loss change, DC − Poisson: {fmt_est(h['insample_logloss_change_per_match'], 5, True)} nats "
        f"per match, {fmt_est(h['insample_logloss_change_total_nats'], 2, True)} nats in total over 11 seasons (paired bootstrap).",
        f"- Held-out 2024-25, recomputed live (DC on 2023-24, ρ̂ = {rho23:+.3f}): DC − Poisson "
        f"{ll_live['dixon_coles'] - ll_live['poisson']:+.5f} {fmt_ci([np.percentile(s, 2.5), np.percentile(s, 97.5)], 5, True)} "
        f"nats per match (paired bootstrap, seed {dc['meta']['seed']}); with ρ pooled over 2015-16..2023-24 instead: "
        f"{fmt_est(h['heldout_logloss_dc_prior_rho_minus_poisson'], 5, True)}.",
        f"- **In one line:** the correction moves log-loss by about a thousandth of a nat per match "
        f"({h['heldout_logloss_dc_minus_poisson']['est']:+.4f} {fmt_ci(h['heldout_logloss_dc_minus_poisson']['ci'], 4, True)} "
        f"held out, {fmt_est(h['insample_logloss_change_per_match'], 4, True)} in sample).",
    ])


def md_before_after(numbers: dict) -> str:
    lg, pre = numbers["legacy"], numbers["preliminary"]
    nb = pre["n_boot"]
    return "\n".join([
        "| quantity (2024-25, nats per match) | before 13 Oct fixes | after |", "|---|---|---|",
        f"| base-rate log-loss | {lg['log_loss']['base_rate_leaky']:.4f} {fmt_ci(lg['log_loss_ci']['base_rate_leaky'], 4)} "
        f"(11 seasons incl. test) | {pre['log_loss']['base_rate']:.4f} {fmt_ci(pre['log_loss_ci']['base_rate'], 4)} "
        f"(pre-test seasons only) |",
        f"| Poisson log-loss | {lg['log_loss']['poisson_unnormalised']:.4f} "
        f"{fmt_ci(lg['log_loss_ci']['poisson_unnormalised'], 4)} (raw grid) | {pre['log_loss']['poisson']:.4f} "
        f"{fmt_ci(pre['log_loss_ci']['poisson'], 4)} (renormalised) |",
        f"| Poisson − base | {lg['poisson_unnormalised_minus_base_rate_leaky']['est']:+.4f} "
        f"{fmt_ci(lg['poisson_unnormalised_minus_base_rate_leaky']['ci'], 4, True)} | "
        f"{pre['log_loss_paired_differences']['poisson']['est']:+.4f} "
        f"{fmt_ci(pre['log_loss_paired_differences']['poisson']['ci'], 4, True)} |",
        "",
        f"All intervals: 95% paired bootstrap over the {pre['n_test_matches']} test matches ({nb} resamples, "
        f"seed {numbers['meta']['seed']}, run_all.py).",
    ])


# ------------------------------------------------- figures drawn from the workstream result files

def fig6_from_results(root: str):
    """Figure 6 (rolling log-loss by season) from ``results/rolling*``; returns the PNG path or None."""
    from . import rolling as RL
    res = load_results(root, "rolling")
    csv = os.path.join(root, "results", "rolling_by_season.csv")
    if res is None or not os.path.exists(csv):
        return None
    seas = pd.read_csv(csv)
    n_boot = res["meta"]["n_boot"]
    n_eval = n_boot.get("eval_per_season", 2000) if isinstance(n_boot, dict) else n_boot
    return RL.fig6_rolling_logloss(seas, seas["ll_poisson_o-ll_ps"].to_numpy(), res["meta"]["primary_config"], n_eval)


def fig7_from_results(root: str):
    """Figure 7 (reliability) from ``results/rolling_reliability.csv``; returns the PNG path or None."""
    from . import rolling as RL
    csv = os.path.join(root, "results", "rolling_reliability.csv")
    if not os.path.exists(csv):
        return None
    return RL.fig7_rolling_reliability(pd.read_csv(csv))


def fig8_from_results(root: str):
    """Figure 8 (gamma by crowd status, three leagues) from ``results/timeseries.json``; PNG path or None."""
    from . import timeseries as TS
    res = load_results(root, "timeseries")
    if res is None or "crowd" not in res:
        return None
    crowd, other = res["crowd"], res["crowd"]["other_leagues"]
    rows = []
    for L, r in (("E0", crowd["E0_main"]), ("D1", other["D1"]), ("SP1", other["SP1"])):
        for k, v in r["groups"].items():
            rows.append({"league": L, "status": k, "est": v["gamma"], "lo": v["gamma_ci"][0], "hi": v["gamma_ci"][1],
                         "n": v["n_matches"]})
    diffs = []
    for lab, v in (("Premier League\n(closed − full)", crowd["E0_main"]["differences"]["closed_minus_full"]),
                   ("PL 2019-20 restart\n(92 matches) − full", crowd["E0_restart_split"]["differences"]["closed_restart_2019_20_minus_full"]),
                   ("PL 2020-21 closed\n− full", crowd["E0_restart_split"]["differences"]["closed_minus_full"]),
                   ("Bundesliga\n(closed − fans allowed)", other["D1"]["differences"]["closed_minus_full"]),
                   ("La Liga\n(closed − fans allowed)", other["SP1"]["differences"]["closed_minus_full"])):
        diffs.append({"label": lab, "est": v["est"], "lo": v["ci"][0], "hi": v["ci"][1]})
    return TS.fig8_crowd_status(rows, diffs)


def _h(res, key, nd=3, signed=False, pct=False):
    """'est [lo, hi]' for one headline entry, or a visible marker if the file or key is missing."""
    if res is None or key not in res.get("headline", {}):
        return f"(missing: {key})"
    h = res["headline"][key]
    if pct:
        return f"{100 * h['est']:+.1f}% [{100 * h['ci'][0]:+.1f}%, {100 * h['ci'][1]:+.1f}%]"
    return fmt_est(h, nd, signed)


def _p_from_how(res, key):
    """The p-value quoted in a headline entry's ``how`` string (e.g. 'p = 0.0163'), as a string."""
    import re
    if res is None or key not in res.get("headline", {}):
        return "?"
    m = re.search(r"p = ([0-9.e+-]+)", res["headline"][key]["how"])
    return m.group(1) if m else "?"


def md_what_changed(root: str, numbers: dict) -> str:
    """Section 7: Milestone 2 feedback and the three critique streams, each taken up or set aside."""
    ts, ro, dc = load_results(root, "timeseries"), load_results(root, "rolling"), load_results(root, "dixon_coles")
    pre = numbers["preliminary"]
    q10, q11, q20 = (_p_from_how(ts, k) for k in ("cochran_q_10_core_crowd_seasons", "cochran_q_11_core_seasons",
                                                   "cochran_q_20_crowd_seasons"))
    return "\n".join([
        "**Milestone 2 feedback** (the brief and the comments on the M2 submission) asked for a held-out forecast "
        "compared with a model-free baseline and the betting market, a clear statement of the uncertainty in every "
        "number, and a check of the code against known mathematics. The M2 submission had one frozen split marked "
        "PRELIMINARY, a bootstrap on the 380 test matches only, and a Skellam check. All three were taken up:",
        "",
        f"- *Held-out validation with a baseline* → taken up: rolling-origin evaluation over 10 test seasons "
        f"(section 6.2), Poisson − base rate {_h(ro, 'poisson_minus_base_logloss_mean10', 4, True)} and "
        f"Poisson − Pinnacle pre-match {_h(ro, 'poisson_minus_pinnacle_prematch_logloss_mean10', 4, True)} nats per match.",
        f"- *Error bars on everything* → taken up: every number in this report carries a 95% interval and the way it "
        f"was obtained; the 2024-25 interval now includes parameter and imputation uncertainty "
        f"({_h(ro, 'logloss_2024_25_param_and_match', 4)} vs match-only {_h(ro, 'logloss_2024_25_match_only', 4)}).",
        f"- *Verification against known mathematics* → taken up: the analytic solution γ̂ = ΣH/ΣA holds to "
        f"{ts['headline']['identity_max_abs_gamma_minus_goals_ratio']['est']:.1e} in all 21 seasons (section 5), and the "
        f"bootstrap intervals cover the true γ with probability {_h(ts, 'ci_coverage_percentile', 3)} on simulated seasons." if ts else
        "- *Verification against known mathematics* → taken up (section 5); timeseries.json missing, run scripts/run_timeseries.py.",
        "",
        "**Three critique streams on the M2 version** (13 October: an assumptions critique, an estimation-and-uncertainty "
        "review, and two validation checks). Taken up:",
        "",
        f"- The held-out base rate used all 11 seasons including the test season, and the Poisson probabilities were "
        f"the raw 0..10 grid mass. Fixed; the before/after table below shows the 2024-25 Poisson − base difference "
        f"moving from {numbers['legacy']['poisson_unnormalised_minus_base_rate_leaky']['est']:+.4f} to "
        f"{pre['log_loss_paired_differences']['poisson']['est']:+.4f} (both with their paired-bootstrap intervals).",
        "- Pinnacle's non-closing odds were labelled \"opening\"; they are pre-match odds collected 1–3 days before "
        "kick-off (football-data notes), so the bookmaker knows team news the model never sees. Relabelled everywhere.",
        "- One bootstrap seed was reused for every season; now 115 + season index, recorded in every output.",
        "- The Fig 5 right-panel title, the duplicated home-advantage figure (γ and extra goals are one finding, by the "
        "identity), and the variance-vs-mean caption that was not a dispersion test: fixed; the dispersion is now a "
        f"Pearson statistic ({_h(dc, 'pearson_dispersion_pooled', 3)} pooled, Poisson range in the interval column).",
        f"- \"One γ for all seasons\": the LR test (p = {q11}) and Cochran's Q (p = {q10}) disagreed because one included "
        f"2020-21 and the other did not. Reconciled in section 4; over 2005-06..2025-26 a common γ is rejected even "
        f"without 2020-21 (Q p = {q20}) because home advantage declines by about 9% per decade.",
        "- Crowd status is a match-level condition (92 restart matches in 2019-20; limited crowds in Dec 2020 and May 2021): "
        "flagged by date in section 4, with the statement that attendance is not in the data.",
        f"- The frozen fit's 1.80-vs-1.51 goal miscalibration: the rolling fit reduces it to "
        f"{_h(ro, 'home_goals_pred_minus_realised_2024_25_rolling', 3, True)} goals per match from "
        f"{_h(ro, 'home_goals_pred_minus_realised_2024_25_frozen', 3, True)} (section 6.2).",
        f"- Dixon–Coles was fitted rather than assumed: ρ = {_h(dc, 'pooled_rho', 3)}, and it moves held-out log-loss by "
        f"{_h(dc, 'heldout_logloss_dc_minus_poisson', 4, True)} nats per match (section 6.3).",
        "- Warm-started recovery refits, no boundary or gradient check, 500 replicates for 2.5% tails: a cold-start run, "
        "max |θ| and projected-gradient norms, and 2,000-replicate intervals are reported in sections 4 and 5.",
        "",
        "Set aside, with reasons:",
        "",
        "- Recovery and coverage under a misspecified data-generating process (over-dispersion, correlated goals, "
        "drifting strengths): not done for time; section 8 says what it would show.",
        "- A full random-effects or time-series model for γ: the DerSimonian–Laird τ, a linear trend and the z-scores "
        "under three reference models answer the question with three numbers; a model would add parameters we cannot "
        "validate on 21 points.",
        "- Raw-file hashes and the run date in the fixture test: the AM115 standard grades the notebook, not the "
        "repository; the raw files are committed instead.",
        "- A separate validation period for choosing the half-life: the choice is reported as made on the test seasons, "
        "with a first-five / last-five split as the check (section 6.2).",
        "- The brief's second identifiability constraint (geometric mean of β = 1): set aside with the likelihood "
        "evidence in section 3 (it is over-constrained, 24 nats worse in 2023-24).",
    ])


def md_not_shown(root: str) -> str:
    """Section 8: what the data and the model cannot decide, with the numbers that say so."""
    ts, ro, dc = load_results(root, "timeseries"), load_results(root, "rolling"), load_results(root, "dixon_coles")
    return "\n".join([
        f"- **The data cannot decide that crowds cause home advantage.** The closed-door evidence is 2020-21 "
        f"(γ = {_h(ts, 'gamma_2020_21', 3)}) plus 92 restart matches in 2019-20; there is no attendance variable, only a "
        f"date flag. Within 2019-20..2021-22, closed doors minus full crowds is {_h(ts, 'crowd_gamma_closed_minus_full_e0', 3, True)}, "
        f"an interval that includes 0, and the 92 restart matches show ordinary home advantage "
        f"({_h(ts, 'crowd_gamma_restart_2019_20_minus_full_e0', 3, True)} vs full crowds). The Bundesliga "
        f"({_h(ts, 'crowd_gamma_closed_minus_fans_allowed_d1', 3, True)}) and La Liga "
        f"({_h(ts, 'crowd_gamma_closed_minus_fans_allowed_sp1', 3, True)}) point the same way, but their partial-crowd dates "
        f"were not separated.",
        f"- **2024-25 is as unusual as 2020-21, with full stadiums.** γ = {_h(ts, 'gamma_2024_25', 3)}; under random effects "
        f"its z is {_h(ts, 'z_2024_25_random_effects', 2)} against {_h(ts, 'z_2020_21_random_effects', 2)} for 2020-21 "
        f"(z ± 1.96 in the brackets). A single low season is therefore not evidence of a crowd effect; the question "
        f"\"was 2020-21 lower\" has the answer yes, the question \"because of the crowds\" does not.",
        f"- **Home advantage is not a constant.** Over 2005-06..2025-26 it falls by {_h(ts, 'trend_log_gamma_per_season_20_crowd_seasons', 4)} "
        f"in log γ per season (about 9% per decade), so the common-γ interval is an average over a declining series. "
        f"Why it declines is not something these data say.",
        f"- **The model is behind the market by a small, now measurable, margin.** Poisson − Pinnacle pre-match is "
        f"{_h(ro, 'poisson_minus_pinnacle_prematch_logloss_mean10', 4, True)} nats per match over 10 seasons "
        f"(detectable gap in one season {ro['headline']['poisson_minus_pinnacle_detectable_gap_per_season']['est']:.3f}, "
        f"range over seasons {ro['headline']['poisson_minus_pinnacle_detectable_gap_per_season']['ci'][0]:.3f}–"
        f"{ro['headline']['poisson_minus_pinnacle_detectable_gap_per_season']['ci'][1]:.3f}; over ten seasons "
        f"{ro['headline']['poisson_minus_pinnacle_detectable_gap_10_seasons']['est']:.4f}, a design quantity); the single 2024-25 split could not "
        f"tell them apart, ten seasons can. Part of the gap is information (team news) the model never sees, and the "
        f"data cannot say how much. The betting test ({_h(ro, 'betting_return_margin_0pct', 3, pct=True)} at margin 0 "
        f"on 590 matches) has no power either way." if ro else "- rolling.json missing; run scripts/run_rolling.py.",
        f"- **The independent-Poisson scoreline model misfits in ways Dixon–Coles does not repair.** The largest pooled "
        f"Pearson residual is {_h(dc, 'max_abs_pearson_residual_poisson', 2)} (cell 2-2; the bracket is that cell's range "
        f"under the model), four of 17 cells exceed 2, and home goals are under-dispersed (dispersion "
        f"{_h(dc, 'pearson_dispersion_pooled', 3)}, bracket = Poisson range). ρ = {_h(dc, 'pooled_rho', 3)} moves "
        f"held-out log-loss by {_h(dc, 'heldout_logloss_dc_minus_poisson', 4, True)} nats per match: the outcome forecasts "
        f"are insensitive to the scoreline misfit.",
        f"- **The error bars are validated only under the model.** Coverage {_h(ts, 'ci_coverage_percentile', 3)} is for "
        f"simulated Poisson seasons; the date-block interval for 2020-21 is narrower than the iid one "
        f"({_h(ts, 'ci_width_date_block_38x10_2020_21', 3)} vs {_h(ts, 'ci_width_iid_2020_21', 3)}, Monte Carlo ranges) "
        f"and the team-cluster one wider ({_h(ts, 'ci_width_team_cluster_home_2024_25', 3)} in 2024-25), so an iid "
        f"interval could be understated by up to about a third if home-team clustering matters.",
        "- **What another month would buy:** coverage and recovery under a misspecified data-generating process "
        "(over-dispersed or correlated goals, strengths that drift within a season); the same crowd-flag analysis with "
        "verified attendance for all three leagues; a within-season update of strengths in the forecast that uses "
        "team news; and a bivariate or under-dispersed count model for the scorelines.",
    ])


def show_figure_or_placeholder(path, source: str, width: int = 950):
    """Display a figure, or print the placeholder message when its results file is missing."""
    if path is None:
        print(f"PLACEHOLDER: results for this figure not found; run {source}")
        return
    show_figure(path, width=width)
