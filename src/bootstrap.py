"""Nonparametric bootstrap of the Poisson fit (resample matches with replacement).

Each replicate is a *weighted* refit: the resample is represented by integer
multiplicities, so no re-indexing is needed and the full-sample estimate is used as
a warm start.
"""
from __future__ import annotations

import numpy as np

from .model import Design, Fit, Spec, fit_poisson, home_advantage_goals, rates

SEED = 115


def bootstrap_fit(design: Design, fit: Fit, n_boot: int = 500, seed: int = SEED,
                  stratify=None, verbose: bool = False) -> dict:
    """Bootstrap gamma and the home-advantage-in-goals statistic.

    ``stratify`` (optional, int array per match) resamples within strata (e.g. within
    season for a pooled fit).  Returns percentile 95% intervals.
    """
    rng = np.random.default_rng(seed)
    spec = Spec(design, fit.constraint)
    m = design.n_matches
    gammas = np.empty(n_boot)
    ha = np.empty(n_boot)
    converged = 0
    strata = np.zeros(m, int) if stratify is None else np.asarray(stratify)
    for b in range(n_boot):
        w = np.zeros(m)
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            draw = rng.choice(idx, size=len(idx), replace=True)
            w += np.bincount(draw, minlength=m)
        fb = fit_poisson(design, weights=w, x0=fit.theta, spec=spec)
        gammas[b] = fb.gamma
        ha[b] = home_advantage_goals(fb, design, weights=w)
        converged += fb.converged
        if verbose and (b + 1) % 100 == 0:
            print(f"    bootstrap {b + 1}/{n_boot}")
    return {
        "n_boot": n_boot,
        "gamma_samples": gammas,
        "ha_goals_samples": ha,
        "gamma_ci": [float(np.percentile(gammas, 2.5)), float(np.percentile(gammas, 97.5))],
        "gamma_se": float(gammas.std(ddof=1)),
        "ha_goals_ci": [float(np.percentile(ha, 2.5)), float(np.percentile(ha, 97.5))],
        "ha_goals_se": float(ha.std(ddof=1)),
        "n_converged": int(converged),
    }


def bootstrap_metrics(prob_sets: dict, outcome_idx: np.ndarray, metric, n_boot: int = 2000,
                      seed: int = SEED, rng=None) -> dict:
    """Paired bootstrap (same resampled matches for every forecaster) of a per-match metric.

    ``metric(probs, outcome_idx)`` must return a scalar mean over matches.  Pass ``rng``
    (a numpy Generator) to control the random stream directly; otherwise one is made from ``seed``.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    m = len(outcome_idx)
    out = {k: np.empty(n_boot) for k in prob_sets}
    for b in range(n_boot):
        idx = rng.integers(0, m, m)
        for k, p in prob_sets.items():
            out[k][b] = metric(p[idx], outcome_idx[idx])
    return {k: {"ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                "se": float(v.std(ddof=1)), "samples": v} for k, v in out.items()}
