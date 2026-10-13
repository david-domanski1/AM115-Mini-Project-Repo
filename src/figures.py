"""All figures, one function per figure.  Each saves PNG (300 dpi) and PDF to figures/."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
COVID = "2020-21"

# One restrained scheme: blue carries the data, orange is reserved for the 2020-21
# highlight, grey for neutral / secondary marks.
C = {"blue": "#2a78d6", "blue_light": "#a9c8ee", "orange": "#eb6834", "orange_light": "#fbd9cb",
     "grey": "#8a8987", "grey_light": "#cfcecb", "aqua": "#1baf7a", "ink": "#0b0b0b", "ink2": "#52514e"}


def setup_style():
    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.titlesize": 16, "legend.fontsize": 12, "figure.titlesize": 17,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["ink2"],
        "axes.labelcolor": C["ink"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 100,
    })


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"))
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    plt.close(fig)
    return os.path.join(FIG_DIR, f"{name}.png")


def _mark_covid(ax, seasons, label="2020-21: empty stadiums", y=None):
    if COVID in seasons:
        i = seasons.index(COVID)
        ax.axvspan(i - 0.5, i + 0.5, color=C["orange_light"], zorder=0, lw=0)
        ax.annotate(label, xy=(i, y if y is not None else ax.get_ylim()[1]), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=12, color=C["orange"])


def fig1_baselines(freq, pooled, name="fig1_baselines"):
    """Grouped bar chart of home/draw/away frequency by season.  ``freq`` indexed by season with H, D, A, n."""
    seasons = list(freq.index)
    x = np.arange(len(seasons))
    w = 0.27
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_ylim(0, 0.62)
    _mark_covid(ax, seasons, y=0.47)
    cols = {"H": C["blue"], "D": C["grey"], "A": C["aqua"]}
    labels = {"H": "Home win", "D": "Draw", "A": "Away win"}
    for k, off in zip("HDA", (-w, 0, w)):
        ax.bar(x + off, freq[k].values, width=w - 0.02, color=cols[k], label=labels[k], zorder=2)
    ax.axhline(pooled["H"], color=C["blue"], ls="--", lw=1.2, zorder=1)
    ax.text(len(seasons) - 0.45, pooled["H"] + 0.008, f"pooled home {pooled['H']:.1%}", ha="right",
            va="bottom", fontsize=12, color=C["blue"])
    ax.set_xticks(x)
    ax.set_xticklabels(seasons, rotation=45, ha="right")
    ax.set_ylabel("Share of matches")
    ax.set_xlabel("Season")
    n = int(freq["n"].min())
    ax.set_title(f"Match outcomes by season, Premier League ({n} matches per season, "
                 f"{int(freq['n'].sum())} total)")
    ax.legend(ncol=3, loc="upper right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    return save(fig, name)


def fig2_home_advantage(table, pooled=None, kind="goals", n_boot=500, name="fig2_home_advantage"):
    """Point estimates with 95% bootstrap CIs by season.

    ``table``: DataFrame indexed by season with columns est, lo, hi for the chosen kind.
    ``pooled``: optional dict(est, lo, hi) for the shared-gamma fit, drawn as a dashed line.
    """
    seasons = list(table.index)
    x = np.arange(len(seasons))
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ref = 0.0 if kind == "goals" else 1.0
    ymin = min(table["lo"].min(), ref) - (0.06 if kind == "goals" else 0.05)
    ymax = max(table["hi"].max(), ref) + (0.10 if kind == "goals" else 0.10)
    ax.set_ylim(ymin, ymax)
    _mark_covid(ax, seasons, y=ymax - 0.04 * (ymax - ymin))
    if pooled is not None:
        ax.axhspan(pooled["lo"], pooled["hi"], color=C["blue_light"], alpha=0.5, lw=0, zorder=0)
        ax.axhline(pooled["est"], color=C["blue"], ls="--", lw=1.2, zorder=1)
    ax.axhline(ref, color=C["ink2"], lw=1.4, zorder=1)
    ax.text(len(seasons) - 0.55, ref, "no home advantage", va="bottom", ha="right", fontsize=12, color=C["ink2"])
    colors = [C["orange"] if s == COVID else C["blue"] for s in seasons]
    for xi, s, c in zip(x, seasons, colors):
        r = table.loc[s]
        ax.errorbar(xi, r["est"], yerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], fmt="o", color=c,
                    ms=9, capsize=5, lw=2, zorder=3)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color=C["blue"], ms=9, lw=2, label="season estimate, 95% CI"),
               Line2D([], [], marker="o", color=C["orange"], ms=9, lw=2, label="2020-21 (empty stadiums)")]
    if pooled is not None:
        handles.append(Line2D([], [], color=C["blue"], ls="--", lw=1.2,
                              label=f"pooled, one γ for all seasons: {pooled['est']:.3f} [{pooled['lo']:.3f}, {pooled['hi']:.3f}]"))
    ax.legend(handles=handles, loc="lower left")
    ax.set_xticks(x)
    ax.set_xticklabels(seasons, rotation=45, ha="right")
    ax.set_xlabel("Season")
    if kind == "goals":
        ax.set_ylabel("Extra home goals per match, λ − λ/γ")
        ax.set_title(f"Home advantage in expected goals, Poisson MLE per season\n"
                     f"(380 matches per season; 95% CI from {n_boot} bootstrap resamples)")
    else:
        ax.set_ylabel("Home-advantage multiplier γ̂")
        ax.set_title(f"Home-advantage multiplier γ̂, Poisson MLE per season\n"
                     f"(380 matches per season; 95% CI from {n_boot} bootstrap resamples)")
    return save(fig, name)


def fig3_goals_vs_poisson(x, y, lam, mu, season, kmax=7, name="fig3_goals_vs_poisson"):
    """Observed goals-per-team histograms vs the fitted per-match Poisson mixture."""
    from scipy.stats import poisson
    k = np.arange(kmax + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, goals, rate, title in zip(axes, (x, y), (lam, mu), ("Home goals", "Away goals")):
        obs = np.bincount(np.minimum(goals, kmax), minlength=kmax + 1) / len(goals)
        exp_pmf = poisson.pmf(k[None, :], rate[:, None]).mean(axis=0)
        exp_pmf[-1] += 1 - exp_pmf.sum()  # fold the tail into the last bin like the observed data
        ax.bar(k, obs, width=0.7, color=C["blue_light"], label="observed", zorder=2)
        ax.plot(k, exp_pmf, "o-", color=C["blue"], ms=8, lw=2, label="fitted Poisson mixture", zorder=3)
        ax.set_xticks(k)
        ax.set_xticklabels([str(i) for i in k[:-1]] + [f"{kmax}+"])
        ax.set_xlabel("Goals scored by the team in a match")
        ax.set_title(f"{title}: mean {goals.mean():.2f}, variance {goals.var(ddof=1):.2f}")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("Share of team-matches")
    fig.suptitle(f"Goals per team per match vs fitted Poisson, {season} (380 matches, 760 team-matches)")
    fig.tight_layout()
    return save(fig, name)


def fig4_parameter_recovery(rec, name="fig4_parameter_recovery"):
    """Fitted vs true log-strengths pooled over simulations, and the distribution of gamma-hat."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    ax = axes[0]
    ta = np.tile(rec["true_log_alpha"], (rec["n_sims"], 1)).ravel()
    tb = np.tile(rec["true_log_beta"], (rec["n_sims"], 1)).ravel()
    ax.scatter(ta, rec["fitted_log_alpha"].ravel(), s=10, alpha=0.25, color=C["blue"], label="log α (attack)", lw=0)
    ax.scatter(tb, rec["fitted_log_beta"].ravel(), s=10, alpha=0.25, color=C["grey"], label="log β (defence)", lw=0)
    lo = min(ta.min(), tb.min(), rec["fitted_log_alpha"].min(), rec["fitted_log_beta"].min()) - 0.1
    hi = max(ta.max(), tb.max(), rec["fitted_log_alpha"].max(), rec["fitted_log_beta"].max()) + 0.1
    ax.plot([lo, hi], [lo, hi], color=C["ink"], lw=1.2, ls="--", label="45° line")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("True log-strength")
    ax.set_ylabel("Fitted log-strength")
    ax.set_title(f"Strengths: RMSE log α = {rec['log_alpha_rmse']:.3f}, log β = {rec['log_beta_rmse']:.3f}")
    ax.legend(loc="upper left", markerscale=3)
    ax = axes[1]
    ax.hist(rec["fitted_gamma"], bins=25, color=C["blue_light"], edgecolor="white")
    ax.axvline(rec["true_gamma"], color=C["orange"], lw=2.5, label=f"true γ = {rec['true_gamma']:.3f}")
    ax.axvline(rec["fitted_gamma"].mean(), color=C["blue"], lw=2, ls="--",
               label=f"mean γ̂ = {rec['fitted_gamma'].mean():.3f}")
    ax.set_xlabel("Fitted home-advantage multiplier γ̂")
    ax.set_ylabel("Number of simulated seasons")
    ax.set_title(f"γ̂: bias = {rec['gamma_bias']:+.4f}, RMSE = {rec['gamma_rmse']:.4f}")
    ax.legend(loc="upper right")
    fig.suptitle(f"Parameter recovery on {rec['n_sims']} simulated seasons of 380 matches "
                 f"(truth = fitted {rec.get('truth_label', '2023-24')} parameters)")
    fig.tight_layout()
    return save(fig, name)


def fig5_logloss_comparison(rows, diffs, season, n, n_boot, name="fig5_logloss_comparison"):
    """Left: log-loss bars with bootstrap CIs.  Right: paired differences.

    ``rows``: list of (label, est, lo, hi); ``diffs``: list of (label, est, lo, hi) of
    paired log-loss differences, each labelled "A − B" (negative = A better).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    labels = [r[0] for r in rows]
    est = np.array([r[1] for r in rows])
    lo = np.array([r[2] for r in rows])
    hi = np.array([r[3] for r in rows])
    cols = [C["grey"], C["blue"], C["aqua"]][:len(rows)]
    xs = np.arange(len(rows))
    ax.bar(xs, est, color=cols, width=0.6, zorder=2)
    ax.errorbar(xs, est, yerr=[est - lo, hi - est], fmt="none", ecolor=C["ink"], capsize=6, lw=1.8, zorder=3)
    for xi, e in zip(xs, est):
        ax.text(xi, e / 2, f"{e:.3f}", ha="center", va="center", fontsize=14, color="white", fontweight="bold")
    ax.axhline(np.log(3), color=C["ink2"], ls=":", lw=1.2)
    ax.text(len(rows) - 0.6, np.log(3) + 0.012, "uniform 1/3 forecast: ln 3 = 1.099", ha="right", fontsize=11, color=C["ink2"])
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Log-loss (nats per match), lower is better")
    ax.set_title(f"Held-out {season}, n = {n} matches")
    ax = axes[1]
    dl = [d[0] for d in diffs]
    de = np.array([d[1] for d in diffs]); dlo = np.array([d[2] for d in diffs]); dhi = np.array([d[3] for d in diffs])
    ys = np.arange(len(diffs))[::-1]
    ax.errorbar(de, ys, xerr=[de - dlo, dhi - de], fmt="o", color=C["blue"], ms=9, capsize=6, lw=2)
    ax.axvline(0, color=C["ink2"], lw=1.2)
    ax.set_yticks(ys); ax.set_yticklabels(dl)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.set_xlabel("Log-loss difference (nats per match)")
    ax.set_title("Paired differences (negative = first forecast better)")
    fig.suptitle(f"PRELIMINARY: 2023-24 strengths forecasting {season}; 95% CIs from {n_boot} paired "
                 f"bootstrap resamples of the {n} held-out matches", fontsize=15)
    fig.tight_layout()
    return save(fig, name)
