"""Synthetic seasons from known parameters, and the parameter-recovery study."""
from __future__ import annotations

import numpy as np

from .model import Design, Spec, fit_poisson, rates

SEED = 115


def simulate_scores(log_alpha, log_beta, log_gamma, home, away, rng):
    lam, mu = rates(np.asarray(log_alpha), np.asarray(log_beta), log_gamma, home, away)
    return rng.poisson(lam), rng.poisson(mu)


def recovery_study(design: Design, log_alpha, log_beta, log_gamma, n_sims: int = 200,
                   seed: int = SEED, constraint: str = "alpha", verbose: bool = False, rng=None) -> dict:
    """Simulate ``n_sims`` seasons on the fixture list of ``design`` and refit each.

    The planted log-alpha must satisfy the same constraint as the fit (mean 0 within
    each group) for the comparison to be meaningful.  Pass ``rng`` (a numpy Generator) to
    control the random stream directly; otherwise one is made from ``seed``.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    log_alpha = np.asarray(log_alpha, float)
    log_beta = np.asarray(log_beta, float)
    n = design.n_teams
    fitted_la = np.empty((n_sims, n))
    fitted_lb = np.empty((n_sims, n))
    fitted_lg = np.empty(n_sims)
    conv = 0
    spec = Spec(design, constraint)
    x0 = spec.pack(log_alpha, log_beta, log_gamma)
    for s in range(n_sims):
        x, y = simulate_scores(log_alpha, log_beta, log_gamma, design.home, design.away, rng)
        d = Design(design.teams, design.home, design.away, x, y, design.groups)
        f = fit_poisson(d, x0=x0, spec=spec)
        fitted_la[s], fitted_lb[s], fitted_lg[s] = f.log_alpha, f.log_beta, f.log_gamma
        conv += f.converged
        if verbose and (s + 1) % 50 == 0:
            print(f"    simulation {s + 1}/{n_sims}")
    gam = np.exp(fitted_lg)
    true_gamma = float(np.exp(log_gamma))
    return {
        "n_sims": n_sims,
        "true_log_alpha": log_alpha, "true_log_beta": log_beta, "true_gamma": true_gamma,
        "fitted_log_alpha": fitted_la, "fitted_log_beta": fitted_lb, "fitted_gamma": gam,
        "gamma_bias": float(gam.mean() - true_gamma),
        "gamma_rmse": float(np.sqrt(np.mean((gam - true_gamma) ** 2))),
        "gamma_sd": float(gam.std(ddof=1)),
        "log_alpha_bias": float((fitted_la - log_alpha).mean()),
        "log_alpha_rmse": float(np.sqrt(np.mean((fitted_la - log_alpha) ** 2))),
        "log_beta_bias": float((fitted_lb - log_beta).mean()),
        "log_beta_rmse": float(np.sqrt(np.mean((fitted_lb - log_beta) ** 2))),
        "alpha_rmse": float(np.sqrt(np.mean((np.exp(fitted_la) - np.exp(log_alpha)) ** 2))),
        "beta_rmse": float(np.sqrt(np.mean((np.exp(fitted_lb) - np.exp(log_beta)) ** 2))),
        "n_converged": int(conv),
    }
