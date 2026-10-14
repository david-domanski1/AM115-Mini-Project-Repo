"""Dixon–Coles low-score correction to the independent-Poisson match model.

Dixon & Coles (1997) multiply the independent-Poisson probability of the four low
scorelines by a factor tau that depends on one extra parameter rho:

    P(X = x, Y = y) = tau(x, y; lambda, mu, rho) * Pois(x; lambda) * Pois(y; mu)

    tau = 1 - lambda*mu*rho   for (0, 0)
          1 + lambda*rho      for (0, 1)
          1 + mu*rho          for (1, 0)
          1 - rho             for (1, 1)
          1                   otherwise.

rho < 0 raises 0-0 and 1-1 and lowers 1-0 and 0-1 (more low-score draws than
independence gives); rho > 0 does the opposite; rho = 0 is the Poisson model.  Over the
infinite grid the corrections cancel exactly (-lambda*mu*rho*p00 + lambda*rho*p01 +
mu*rho*p10 - rho*p11 = 0 because p01 = mu*p00, p10 = lambda*p00, p11 = lambda*mu*p00),
so tau re-weights probability between the four cells without changing the total mass
and without changing the home-goal or away-goal marginal means by more than these four
cells allow.  All tau must be non-negative, which for one match means

    max(-1/lambda, -1/mu) <= rho <= min(1/(lambda*mu), 1),

and for a season the intersection of these intervals over all matches
(:func:`rho_admissible_range`).

Estimation: the log-likelihood is the Poisson log-likelihood plus sum_m log tau_m.  The
fit is joint over (log-strengths, log gamma, rho) with an analytic gradient
(:func:`dc_negloglik_and_grad`), warm-started at the Poisson MLE with rho = 0.  The
profile likelihood in rho (:func:`profile_loglik`) refits all other parameters at each
fixed rho and is used for a likelihood-ratio interval and as a cross-check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import chi2

from .model import LOG_BOUNDS, Design, Spec, fit_poisson, outcome_probs, rates, score_grid

RHO_BOUNDS = (-0.5, 0.5)   # optimiser box; admissibility is checked on the fitted rates
OUTCOMES = "HDA"


# --------------------------------------------------------------------------- tau

def tau(x, y, lam, mu, rho):
    """Dixon–Coles factor for scorelines (x, y) with rates (lam, mu); broadcasts."""
    x, y = np.asarray(x), np.asarray(y)
    lam, mu = np.asarray(lam, float), np.asarray(mu, float)
    t = np.ones(np.broadcast(x, y, lam, mu).shape)
    t = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1.0 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1.0 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1.0 - rho, t)
    return t


def rho_admissible_range(lam, mu):
    """(lo, hi) such that every tau >= 0 for every match: max(-1/lam, -1/mu) <= rho <= min(1/(lam mu), 1)."""
    lam, mu = np.atleast_1d(lam), np.atleast_1d(mu)
    lo = float(np.max(np.maximum(-1.0 / lam, -1.0 / mu)))
    hi = float(min(np.min(1.0 / (lam * mu)), 1.0))
    return lo, hi


def dc_grid(lam, mu, rho, max_goals=10):
    """(m, K+1, K+1) score grid under Dixon–Coles: the Poisson grid with the four low cells times tau."""
    g = score_grid(lam, mu, max_goals).copy()
    lam, mu = np.atleast_1d(lam).astype(float), np.atleast_1d(mu).astype(float)
    g[:, 0, 0] *= 1.0 - lam * mu * rho
    g[:, 0, 1] *= 1.0 + lam * rho
    g[:, 1, 0] *= 1.0 + mu * rho
    g[:, 1, 1] *= 1.0 - rho
    return g


def dc_outcome_probs(lam, mu, rho, max_goals=10, normalise=True):
    """P(H), P(D), P(A) under Dixon–Coles, renormalised on the 0..max_goals grid like ``outcome_probs``.

    With rho = 0 this is exactly ``src.model.outcome_probs``.
    """
    if rho == 0:
        return outcome_probs(lam, mu, max_goals=max_goals, normalise=normalise)
    g = dc_grid(lam, mu, rho, max_goals)
    K = max_goals + 1
    i, j = np.indices((K, K))
    p = np.column_stack([g[:, i > j].sum(axis=1), g[:, i == j].sum(axis=1), g[:, i < j].sum(axis=1)])
    if normalise:
        p = p / p.sum(axis=1, keepdims=True)
    return p


# -------------------------------------------------------------------- likelihood

def _log_tau_and_derivs(x, y, lam, mu, rho):
    """log tau per match and its derivatives with respect to log lam, log mu and rho."""
    m = len(x)
    lt, gl, gm, gr = np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m)
    c00 = (x == 0) & (y == 0)
    c01 = (x == 0) & (y == 1)
    c10 = (x == 1) & (y == 0)
    c11 = (x == 1) & (y == 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = 1.0 - lam[c00] * mu[c00] * rho
        lt[c00] = np.log(np.clip(t, 1e-300, None))
        gl[c00] = gm[c00] = -lam[c00] * mu[c00] * rho / t
        gr[c00] = -lam[c00] * mu[c00] / t
        t = 1.0 + lam[c01] * rho
        lt[c01] = np.log(np.clip(t, 1e-300, None))
        gl[c01] = lam[c01] * rho / t
        gr[c01] = lam[c01] / t
        t = 1.0 + mu[c10] * rho
        lt[c10] = np.log(np.clip(t, 1e-300, None))
        gm[c10] = mu[c10] * rho / t
        gr[c10] = mu[c10] / t
        t = 1.0 - rho
        lt[c11] = np.log(max(t, 1e-300))
        gr[c11] = -1.0 / t
    return lt, gl, gm, gr


def dc_negloglik_and_grad(params, design: Design, spec: Spec, weights=None):
    """Negative weighted Dixon–Coles log-likelihood (no factorial terms) and its analytic gradient.

    ``params`` = Poisson free parameters (layout of ``spec``) followed by rho.
    """
    theta, rho = params[:-1], float(params[-1])
    la, lb, lg = spec.unpack(theta)
    lam, mu = rates(la, lb, lg, design.home, design.away)
    x, y = design.x, design.y
    w = np.ones_like(lam) if weights is None else weights
    lt, gl, gm, gr = _log_tau_and_derivs(x, y, lam, mu, rho)
    ll = np.sum(w * (x * np.log(lam) - lam + y * np.log(mu) - mu + lt))
    rx = w * (x - lam + gl)
    ry = w * (y - mu + gm)
    n = design.n_teams
    da = np.bincount(design.home, rx, n) + np.bincount(design.away, ry, n)
    db = np.bincount(design.away, rx, n) + np.bincount(design.home, ry, n)
    grad_b = spec._reduce_grad(db) if spec.constraint == "both" else db
    grad = np.concatenate([spec._reduce_grad(da), grad_b, [rx.sum()], [np.sum(w * gr)]])
    return -ll, -grad


def dc_loglik_full(design: Design, la, lb, lg, rho, weights=None):
    """Full Dixon–Coles log-likelihood including the factorial terms."""
    lam, mu = rates(la, lb, lg, design.home, design.away)
    x, y = design.x, design.y
    w = np.ones(design.n_matches) if weights is None else weights
    lt = np.log(np.clip(tau(x, y, lam, mu, rho), 1e-300, None))
    return float(np.sum(w * (x * np.log(lam) - lam - gammaln(x + 1) + y * np.log(mu) - mu - gammaln(y + 1) + lt)))


@dataclass
class DCFit:
    teams: list
    log_alpha: np.ndarray
    log_beta: np.ndarray
    log_gamma: float
    rho: float
    params: np.ndarray        # Poisson free parameters followed by rho
    loglik: float             # full log-likelihood incl. factorial terms
    n_params: int
    converged: bool
    admissible: tuple         # admissible rho range at the fitted rates
    poisson_loglik: float     # log-likelihood of the Poisson fit it was started from (rho = 0)

    @property
    def gamma(self):
        return float(np.exp(self.log_gamma))

    def rates(self, design: Design):
        return rates(self.log_alpha, self.log_beta, self.log_gamma, design.home, design.away)


def fit_dixon_coles(design: Design, weights=None, x0=None, spec: Spec | None = None,
                    fixed_rho: float | None = None, poisson_fit=None) -> DCFit:
    """Joint MLE of the Poisson strengths, gamma and rho (sign free).

    ``x0`` = warm start for (theta, rho); default is the Poisson MLE with rho = 0.
    ``fixed_rho`` holds rho at a value (used for the profile likelihood).
    """
    spec = spec or Spec(design)
    if poisson_fit is None and (x0 is None):
        poisson_fit = fit_poisson(design, weights=weights, spec=spec)
    if x0 is None:
        x0 = np.concatenate([poisson_fit.theta, [0.0 if fixed_rho is None else fixed_rho]])
    x0 = np.array(x0, float)
    rb = RHO_BOUNDS if fixed_rho is None else (fixed_rho, fixed_rho)
    if fixed_rho is not None:
        x0[-1] = fixed_rho
    res = minimize(dc_negloglik_and_grad, x0, args=(design, spec, weights), jac=True, method="L-BFGS-B",
                   bounds=[LOG_BOUNDS] * spec.n_params + [rb],
                   options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-8})
    la, lb, lg = spec.unpack(res.x[:-1])
    rho = float(res.x[-1])
    lam, mu = rates(la, lb, lg, design.home, design.away)
    p_ll = poisson_fit.loglik if poisson_fit is not None else np.nan
    return DCFit(list(design.teams), la, lb, lg, rho, res.x, dc_loglik_full(design, la, lb, lg, rho, weights),
                 spec.n_params + (fixed_rho is None), bool(res.success), rho_admissible_range(lam, mu), p_ll)


def lr_test(loglik_dc: float, loglik_poisson: float, df: int = 1) -> dict:
    """Likelihood-ratio test of rho = 0: 2 (l_DC - l_Pois) ~ chi^2_df under H0."""
    stat = max(2.0 * (loglik_dc - loglik_poisson), 0.0)
    return {"statistic": float(stat), "df": df, "p_value": float(chi2.sf(stat, df))}


def profile_loglik(design: Design, rho_grid, dc_fit: DCFit, spec: Spec | None = None) -> np.ndarray:
    """Profile log-likelihood: maximise over all other parameters at each fixed rho (warm-started)."""
    spec = spec or Spec(design)
    out = np.empty(len(rho_grid))
    for k, r in enumerate(rho_grid):
        f = fit_dixon_coles(design, x0=dc_fit.params, spec=spec, fixed_rho=float(r))
        out[k] = f.loglik
    return out


def profile_ci(rho_grid, prof, level=0.95):
    """Likelihood-ratio interval {rho : 2 (max - prof(rho)) <= chi2_1(level)} by linear interpolation."""
    rho_grid, prof = np.asarray(rho_grid), np.asarray(prof)
    cut = prof.max() - chi2.ppf(level, 1) / 2.0
    k = int(np.argmax(prof))
    lo_idx = np.flatnonzero(prof[:k + 1] < cut)
    hi_idx = np.flatnonzero(prof[k:] < cut) + k
    lo = hi = np.nan
    if len(lo_idx):
        i = lo_idx[-1]
        lo = float(np.interp(cut, [prof[i], prof[i + 1]], [rho_grid[i], rho_grid[i + 1]]))
    if len(hi_idx):
        i = hi_idx[0]
        hi = float(np.interp(cut, [prof[i], prof[i - 1]], [rho_grid[i], rho_grid[i - 1]]))
    return [lo, hi]


def bootstrap_rho(design: Design, dc_fit: DCFit, n_boot: int, rng, stratify=None, spec: Spec | None = None):
    """Nonparametric bootstrap of rho: resample matches (within strata), refit jointly, warm-started.

    Returns the rho samples, the log-likelihood gains over the Poisson refit are not
    recomputed (only rho is bootstrapped).
    """
    spec = spec or Spec(design)
    m = design.n_matches
    strata = np.zeros(m, int) if stratify is None else np.asarray(stratify)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    rhos = np.empty(n_boot)
    conv = 0
    for b in range(n_boot):
        w = np.zeros(m)
        for idx in groups:
            w += np.bincount(rng.choice(idx, size=len(idx), replace=True), minlength=m)
        f = fit_dixon_coles(design, weights=w, x0=dc_fit.params, spec=spec)
        rhos[b] = f.rho
        conv += f.converged
    return {"samples": rhos, "ci": [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))],
            "se": float(rhos.std(ddof=1)), "n_boot": n_boot, "n_converged": int(conv)}


# ------------------------------------------------------------------ simulation

def simulate_dc_scores(lam, mu, rho, rng, max_goals=10):
    """Draw one scoreline per match from the Dixon–Coles grid (renormalised on 0..max_goals)."""
    g = dc_grid(lam, mu, rho, max_goals)
    m, K = g.shape[0], max_goals + 1
    flat = g.reshape(m, -1)
    cdf = np.cumsum(flat, axis=1)
    cdf /= cdf[:, -1:]
    u = rng.random(m)[:, None]
    k = (u > cdf).sum(axis=1)
    return k // K, k % K


# ------------------------------------------------------------ goodness of fit

SCORE_CELLS = [f"{i}-{j}" for i in range(4) for j in range(4)] + ["4+"]


def scoreline_counts(x, y):
    """Observed counts in the 17 cells: 0..3 x 0..3 and '4+' (either side scores 4 or more)."""
    x, y = np.asarray(x), np.asarray(y)
    c = [int(np.sum((x == i) & (y == j))) for i in range(4) for j in range(4)]
    c.append(int(np.sum((x >= 4) | (y >= 4))))
    return np.array(c)


def scoreline_expected(lam, mu, rho=0.0, max_goals=10):
    """Expected counts in the 17 cells, summed over matches; '4+' = 1 - sum of the 16 low cells per match."""
    g = dc_grid(lam, mu, rho, max_goals) if rho != 0 else score_grid(lam, mu, max_goals)
    low = g[:, :4, :4].reshape(len(g), -1)
    e = low.sum(axis=0)
    return np.append(e, float(np.sum(1.0 - low.sum(axis=1))))


def scoreline_table(x, y, lam_p, mu_p, lam_dc, mu_dc, rho):
    """Pooled scoreline table: observed, expected under Poisson and Dixon–Coles, Pearson residuals."""
    import pandas as pd
    obs = scoreline_counts(x, y)
    ep = scoreline_expected(lam_p, mu_p, 0.0)
    ed = scoreline_expected(lam_dc, mu_dc, rho)
    return pd.DataFrame({"scoreline": SCORE_CELLS, "observed": obs, "expected_poisson": ep,
                         "pearson_poisson": (obs - ep) / np.sqrt(ep), "expected_dc": ed,
                         "pearson_dc": (obs - ed) / np.sqrt(ed)})


def scoreline_null_envelope(lam, mu, rho, n_sim, rng):
    """Parametric bootstrap under the fitted model (no refit): Pearson residuals of the 17 cells.

    Returns an (n_sim, 17) array of residuals and the per-simulation max |residual|.
    Not refitting makes the envelope slightly wider than a refitted one (conservative).
    """
    e = scoreline_expected(lam, mu, rho)
    res = np.empty((n_sim, len(SCORE_CELLS)))
    for s in range(n_sim):
        if rho == 0:
            x, y = rng.poisson(lam), rng.poisson(mu)
        else:
            x, y = simulate_dc_scores(lam, mu, rho, rng)
        res[s] = (scoreline_counts(x, y) - e) / np.sqrt(e)
    return res, np.abs(res).max(axis=1)


def goal_histogram_envelope(x, y, lam, mu, n_sim, rng, kmax=6):
    """Observed goal counts 0..kmax+ (home and away) vs the fitted Poisson mixture and a parametric-bootstrap band.

    Each simulated data set draws X_m ~ Pois(lam_m), Y_m ~ Pois(mu_m) for every match
    (no refit), so the band shows the sampling spread of the histogram if the model
    were true.  Returns a dict with observed, expected, lo, hi for 'home' and 'away'.
    """
    from scipy.stats import poisson
    k = np.arange(kmax + 1)
    out = {"k": k, "n_sim": n_sim, "n_matches": len(x)}
    for side, goals, rate in (("home", x, lam), ("away", y, mu)):
        obs = np.bincount(np.minimum(goals, kmax), minlength=kmax + 1)
        pmf = poisson.pmf(k[None, :], np.asarray(rate)[:, None])
        pmf[:, -1] = 1.0 - pmf[:, :-1].sum(axis=1)
        exp = pmf.sum(axis=0)
        sims = np.empty((n_sim, kmax + 1))
        for s in range(n_sim):
            sims[s] = np.bincount(np.minimum(rng.poisson(rate), kmax), minlength=kmax + 1)
        lo, hi = np.percentile(sims, [2.5, 97.5], axis=0)
        out[side] = {"observed": obs, "expected": exp, "lo": lo, "hi": hi,
                     "outside": [int(i) for i in k if obs[i] < lo[i] or obs[i] > hi[i]]}
    return out


def pearson_dispersion(x, y, lam, mu, n_params):
    """Pearson dispersion sum((x-lam)^2/lam + (y-mu)^2/mu) / (2m - p); 1 under the Poisson model."""
    x, y = np.asarray(x), np.asarray(y)
    stat = float(np.sum((x - lam) ** 2 / lam) + np.sum((y - mu) ** 2 / mu))
    return stat / (2 * len(x) - n_params)


# ------------------------------------------------------------------- figures

def fig3_goals_envelope(env, title_extra="", name="fig3_goals_envelope"):
    """Observed goal histograms (bars) vs the fitted Poisson mixture (line) with a parametric-bootstrap 95% band."""
    import matplotlib.pyplot as plt
    from . import figures as F
    k = env["k"]
    kmax = int(k[-1])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, side, title in zip(axes, ("home", "away"), ("Home goals", "Away goals")):
        e = env[side]
        ax.bar(k, e["observed"], width=0.7, color=F.C["blue_light"], label="observed", zorder=2)
        ax.fill_between(k, e["lo"], e["hi"], color=F.C["grey_light"], alpha=0.8, step=None, zorder=1,
                        label=f"95% band, {env['n_sim']} simulated data sets")
        ax.plot(k, e["expected"], "o-", color=F.C["blue"], ms=7, lw=2, label="fitted Poisson mixture", zorder=3)
        for i in e["outside"]:
            ax.plot(k[i], e["observed"][i], "s", color=F.C["orange"], ms=9, zorder=4)
        ax.set_xticks(k)
        ax.set_xticklabels([str(i) for i in k[:-1]] + [f"{kmax}+"])
        ax.set_xlabel("Goals scored by the team in a match")
        n_out = len(e["outside"])
        ax.set_title(f"{title}: {n_out} of {len(k)} bins outside the band")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("Number of matches")
    fig.suptitle(f"Goals per match vs fitted Poisson, {env['n_matches']} matches{title_extra}")
    fig.tight_layout()
    return F.save(fig, name)


def fig9_dixon_coles(by_season, pooled, rho_grid, prof, table, name="fig9_dixon_coles"):
    """(a) rho by season with bootstrap CIs and the pooled band; (b) pooled profile log-likelihood;
    (c) Pearson residuals of the pooled scoreline table under Poisson and Dixon–Coles."""
    import matplotlib.pyplot as plt
    from . import figures as F
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1.3, 1, 1.3]})
    ax = axes[0]
    seasons = list(by_season["season"])
    xs = np.arange(len(seasons))
    ax.axhspan(pooled["ci"][0], pooled["ci"][1], color=F.C["blue_light"], alpha=0.5, lw=0)
    ax.axhline(pooled["est"], color=F.C["blue"], ls="--", lw=1.2,
               label=f"pooled ρ = {pooled['est']:+.3f} [{pooled['ci'][0]:+.3f}, {pooled['ci'][1]:+.3f}]")
    ax.axhline(0, color=F.C["ink2"], lw=1.2)
    cols = [F.C["orange"] if s == "2020-21" else F.C["blue"] for s in seasons]
    for xi, (_, r), c in zip(xs, by_season.iterrows(), cols):
        ax.errorbar(xi, r["rho"], yerr=[[r["rho"] - r["rho_lo"]], [r["rho_hi"] - r["rho"]]], fmt="o",
                    color=c, ms=8, capsize=4, lw=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(seasons, rotation=45, ha="right")
    ax.set_ylabel("Dixon–Coles ρ")
    ax.set_title("ρ by season, 95% bootstrap CI")
    ax.legend(loc="lower left", fontsize=11)
    ax = axes[1]
    d = 2 * (prof.max() - prof)
    ax.plot(rho_grid, d, color=F.C["blue"], lw=2)
    ax.axhline(3.841, color=F.C["ink2"], ls=":", lw=1.2)
    ax.text(rho_grid[0], 3.841, " χ²₁ 95% cut", va="bottom", fontsize=11, color=F.C["ink2"])
    ax.axvline(0, color=F.C["ink2"], lw=1.2)
    ax.axvline(pooled["est"], color=F.C["blue"], ls="--", lw=1.2)
    ax.set_ylim(0, max(8, float(d[np.argmin(np.abs(rho_grid))]) + 1))
    ax.set_xlabel("ρ (other parameters re-maximised)")
    ax.set_ylabel("2 × (max log-lik − profile log-lik)")
    ax.set_title("Pooled profile likelihood")
    ax = axes[2]
    ys = np.arange(len(table))[::-1]
    ax.axvspan(-2, 2, color=F.C["grey_light"], alpha=0.5, lw=0)
    ax.plot(table["pearson_poisson"], ys, "o", color=F.C["grey"], ms=8, label="Poisson")
    ax.plot(table["pearson_dc"], ys, "D", color=F.C["blue"], ms=6, label="Dixon–Coles")
    ax.axvline(0, color=F.C["ink2"], lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(table["scoreline"], fontsize=10)
    ax.set_xlabel("Pearson residual (obs − exp)/√exp")
    ax.set_title("Pooled scorelines (band = ±2)")
    ax.legend(loc="lower right", fontsize=11)
    fig.suptitle(f"Dixon–Coles low-score correction, {int(table['observed'].sum())} matches, 2015-16..2025-26")
    fig.tight_layout()
    return F.save(fig, name)
