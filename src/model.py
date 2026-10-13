"""Independent-Poisson match model with a common home-advantage multiplier.

    X ~ Poisson(lambda),  Y ~ Poisson(mu),  independent
    lambda = alpha_home * beta_away * gamma,      mu = alpha_away * beta_home

Parameters are estimated on the log scale by maximum likelihood (L-BFGS-B, analytic
gradient).  Identifiability: the likelihood is invariant to alpha -> c*alpha,
beta -> beta/c, and to nothing else, so exactly one constraint is needed.  The
default (``constraint='alpha'``) imposes geometric-mean(alpha) = 1 and leaves beta
free, giving 20 + 20 + 1 - 1 = 40 free parameters per season.  Imposing
geometric-mean(beta) = 1 as well (``constraint='both'``, 39 free parameters) forces
the geometric mean of the away rate mu to be exactly 1 and biases gamma; it is
implemented only as a diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, ive

LOG_BOUNDS = (-5.0, 5.0)


@dataclass
class Design:
    """Integer-encoded fixture list."""
    teams: list                # team labels, index = parameter index
    home: np.ndarray           # int index of home team per match
    away: np.ndarray           # int index of away team per match
    x: np.ndarray              # home goals
    y: np.ndarray              # away goals
    groups: np.ndarray         # constraint group per team (0 for a single season)

    @property
    def n_teams(self):
        return len(self.teams)

    @property
    def n_matches(self):
        return len(self.x)


def make_design(df, home="home", away="away", x="home_goals", y="away_goals", season=None) -> Design:
    """Build a Design from a tidy match table.

    If ``season`` is a column name, each (season, team) pair becomes its own
    parameter and the identifiability constraint is applied within each season.
    """
    if season is None:
        teams = sorted(set(df[home]) | set(df[away]))
        idx = {t: i for i, t in enumerate(teams)}
        h = df[home].map(idx).to_numpy()
        a = df[away].map(idx).to_numpy()
        groups = np.zeros(len(teams), dtype=int)
    else:
        pairs = sorted(set(zip(df[season], df[home])) | set(zip(df[season], df[away])))
        idx = {p: i for i, p in enumerate(pairs)}
        h = np.array([idx[p] for p in zip(df[season], df[home])])
        a = np.array([idx[p] for p in zip(df[season], df[away])])
        seasons = sorted({p[0] for p in pairs})
        groups = np.array([seasons.index(p[0]) for p in pairs])
        teams = pairs
    return Design(teams, h, a, df[x].to_numpy(int), df[y].to_numpy(int), groups)


class Spec:
    """Free-parameter layout for a given design and constraint choice."""

    def __init__(self, design: Design, constraint: str = "alpha"):
        assert constraint in ("alpha", "both")
        self.constraint = constraint
        self.n = design.n_teams
        self.groups = design.groups
        self.n_groups = int(design.groups.max()) + 1
        self.det = np.array([np.flatnonzero(design.groups == g)[-1] for g in range(self.n_groups)])
        self.free = np.ones(self.n, dtype=bool)
        self.free[self.det] = False
        self.n_free_a = self.n - self.n_groups
        self.n_free_b = self.n_free_a if constraint == "both" else self.n
        self.n_params = self.n_free_a + self.n_free_b + 1

    def _expand(self, free_vals):
        v = np.empty(self.n)
        v[self.free] = free_vals
        v[self.det] = -np.bincount(self.groups[self.free], weights=free_vals, minlength=self.n_groups)
        return v

    def _reduce_grad(self, grad_full):
        return grad_full[self.free] - grad_full[self.det][self.groups[self.free]]

    def unpack(self, theta):
        """theta -> (log_alpha[n], log_beta[n], log_gamma)."""
        la = self._expand(theta[:self.n_free_a])
        if self.constraint == "both":
            lb = self._expand(theta[self.n_free_a:self.n_free_a + self.n_free_b])
        else:
            lb = np.asarray(theta[self.n_free_a:self.n_free_a + self.n])
        return la, lb, float(theta[-1])

    def pack(self, log_alpha, log_beta, log_gamma):
        parts = [np.asarray(log_alpha)[self.free]]
        parts.append(np.asarray(log_beta)[self.free] if self.constraint == "both" else np.asarray(log_beta))
        parts.append([log_gamma])
        return np.concatenate(parts)


def rates(log_alpha, log_beta, log_gamma, home, away):
    lam = np.exp(log_alpha[home] + log_beta[away] + log_gamma)
    mu = np.exp(log_alpha[away] + log_beta[home])
    return lam, mu


def negloglik_and_grad(theta, design: Design, spec: Spec, weights=None):
    """Negative weighted Poisson log-likelihood (without factorial terms) and its gradient."""
    la, lb, lg = spec.unpack(theta)
    lam, mu = rates(la, lb, lg, design.home, design.away)
    x, y = design.x, design.y
    w = np.ones_like(lam) if weights is None else weights
    ll = np.sum(w * (x * np.log(lam) - lam + y * np.log(mu) - mu))
    rx = w * (x - lam)
    ry = w * (y - mu)
    n = design.n_teams
    da = np.bincount(design.home, rx, n) + np.bincount(design.away, ry, n)
    db = np.bincount(design.away, rx, n) + np.bincount(design.home, ry, n)
    dg = rx.sum()
    grad_b = spec._reduce_grad(db) if spec.constraint == "both" else db
    grad = np.concatenate([spec._reduce_grad(da), grad_b, [dg]])
    return -ll, -grad


@dataclass
class Fit:
    teams: list
    log_alpha: np.ndarray
    log_beta: np.ndarray
    log_gamma: float
    theta: np.ndarray
    loglik: float          # full log-likelihood incl. factorial terms
    n_params: int
    converged: bool
    n_iter: int
    constraint: str

    @property
    def alpha(self):
        return np.exp(self.log_alpha)

    @property
    def beta(self):
        return np.exp(self.log_beta)

    @property
    def gamma(self):
        return float(np.exp(self.log_gamma))

    def strength(self, team):
        i = self.teams.index(team)
        return self.alpha[i], self.beta[i]


def fit_poisson(design: Design, weights=None, x0=None, constraint: str = "alpha",
                spec: Spec | None = None) -> Fit:
    """Maximum-likelihood fit.  ``weights`` (e.g. bootstrap counts) and ``x0`` (warm start) are optional."""
    spec = spec or Spec(design, constraint)
    if x0 is None:
        x0 = np.zeros(spec.n_params)
        w = np.ones(design.n_matches) if weights is None else weights
        mx, my = np.average(design.x, weights=w), np.average(design.y, weights=w)
        x0[spec.n_free_a:spec.n_free_a + spec.n_free_b] = 0.0 if spec.constraint == "both" else np.log(max(my, 1e-3))
        x0[-1] = np.log(max(mx, 1e-3) / max(my, 1e-3))
    res = minimize(negloglik_and_grad, x0, args=(design, spec, weights), jac=True, method="L-BFGS-B",
                   bounds=[LOG_BOUNDS] * spec.n_params,
                   options={"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-8})
    la, lb, lg = spec.unpack(res.x)
    lam, mu = rates(la, lb, lg, design.home, design.away)
    w = np.ones(design.n_matches) if weights is None else weights
    full_ll = float(np.sum(w * (design.x * np.log(lam) - lam - gammaln(design.x + 1)
                                + design.y * np.log(mu) - mu - gammaln(design.y + 1))))
    return Fit(list(design.teams), la, lb, lg, res.x, full_ll, spec.n_params, bool(res.success),
               int(res.nit), spec.constraint)


def poisson_pmf_matrix(rate, max_goals=10):
    """(m, max_goals+1) matrix of Poisson pmfs, one row per match."""
    k = np.arange(max_goals + 1)
    rate = np.atleast_1d(np.asarray(rate, float))
    return np.exp(k[None, :] * np.log(rate)[:, None] - rate[:, None] - gammaln(k + 1)[None, :])


def score_grid(lam, mu, max_goals=10):
    """(m, K+1, K+1) array with entry [m, i, j] = P(X_m = i, Y_m = j) on the truncated grid."""
    ph = poisson_pmf_matrix(lam, max_goals)
    pa = poisson_pmf_matrix(mu, max_goals)
    return ph[:, :, None] * pa[:, None, :]


def outcome_probs(lam, mu, max_goals=10, normalise=True):
    """P(home win), P(draw), P(away win) by summing the score grid over 0..max_goals goals each.

    Returns an (m, 3) array in the order H, D, A.  With ``normalise=True`` (default) the
    three probabilities are divided by the grid mass so they sum to exactly 1 (this
    conditions on the match ending inside the 0..max_goals grid).  With
    ``normalise=False`` they sum to the raw grid mass, which is < 1 by the truncation
    error (at most about 0.6% for Premier League rates on the 0..10 grid).
    """
    g = score_grid(lam, mu, max_goals)
    K = max_goals + 1
    i, j = np.indices((K, K))
    pH = g[:, i > j].sum(axis=1)
    pD = g[:, i == j].sum(axis=1)
    pA = g[:, i < j].sum(axis=1)
    p = np.column_stack([pH, pD, pA])
    if normalise:
        p = p / p.sum(axis=1, keepdims=True)
    return p


def skellam_pmf(k, lam, mu):
    """P(X - Y = k) for independent Poissons: exp(-(lam+mu)) (lam/mu)^(k/2) I_k(2 sqrt(lam mu)).

    Uses the exponentially-scaled Bessel function ``ive`` for numerical stability.
    """
    lam = np.asarray(lam, float)
    mu = np.asarray(mu, float)
    k = np.asarray(k)
    z = 2.0 * np.sqrt(lam * mu)
    return np.exp(-(lam + mu) + z) * (lam / mu) ** (k / 2.0) * ive(np.abs(k), z)


def skellam_outcome_probs(lam, mu, k_max=60):
    """P(H), P(D), P(A) from the Skellam closed form (goal difference = X - Y)."""
    lam = np.atleast_1d(np.asarray(lam, float))
    mu = np.atleast_1d(np.asarray(mu, float))
    ks = np.arange(1, k_max + 1)
    pD = skellam_pmf(0, lam, mu)
    pH = skellam_pmf(ks[None, :], lam[:, None], mu[:, None]).sum(axis=1)
    pA = skellam_pmf(-ks[None, :], lam[:, None], mu[:, None]).sum(axis=1)
    return np.column_stack([pH, pD, pA])


def home_advantage_goals(fit: Fit, design: Design, weights=None):
    """Mean over matches of lambda - lambda/gamma: extra home goals produced by the multiplier."""
    lam, _ = rates(fit.log_alpha, fit.log_beta, fit.log_gamma, design.home, design.away)
    extra = lam - lam / fit.gamma
    return float(np.average(extra, weights=weights))
