"""Dixon–Coles correction: mass conservation, rho = 0 limit, gradient, and recovery of rho."""
import numpy as np

from src.dixon_coles import (dc_grid, dc_negloglik_and_grad, dc_outcome_probs, fit_dixon_coles, rho_admissible_range,
                             scoreline_counts, scoreline_expected, simulate_dc_scores, bootstrap_rho)
from src.model import Design, Spec, fit_poisson, outcome_probs
from src.report import gradient_check

rng = np.random.default_rng(115)
LAM = rng.uniform(0.4, 3.0, 200)
MU = rng.uniform(0.4, 2.5, 200)


def _season(n_teams=20, rng=None):
    """Double round robin with planted strengths."""
    h, a = np.array([(i, j) for i in range(n_teams) for j in range(n_teams) if i != j]).T
    la = rng.normal(0, 0.25, n_teams)
    la -= la.mean()
    lb = rng.normal(0.1, 0.2, n_teams)
    return h, a, la, lb


def test_tau_preserves_mass_for_admissible_rho():
    lo, hi = rho_admissible_range(LAM, MU)
    assert lo < 0 < hi
    for rho in (lo * 0.99, -0.1, -0.03, 0.05, hi * 0.99):
        g = dc_grid(LAM, MU, rho, max_goals=40)
        assert (g >= 0).all()
        assert np.abs(g.sum(axis=(1, 2)) - 1).max() < 1e-10


def test_rho_zero_reproduces_outcome_probs():
    assert np.array_equal(dc_outcome_probs(LAM, MU, 0.0), outcome_probs(LAM, MU))
    g = dc_grid(LAM, MU, 1e-300)
    p = np.column_stack([np.tril(gi, -1).sum() for gi in g])
    assert np.allclose(p.ravel(), outcome_probs(LAM, MU, normalise=False)[:, 0], atol=1e-14)


def test_dc_raises_low_draws_for_negative_rho():
    p0, pn = outcome_probs(LAM, MU), dc_outcome_probs(LAM, MU, -0.1)
    assert (pn[:, 1] > p0[:, 1]).all()
    assert np.allclose(pn.sum(axis=1), 1)


def test_scoreline_cells_sum_to_matches():
    x, y = rng.poisson(LAM), rng.poisson(MU)
    assert scoreline_counts(x, y).sum() == len(x)
    assert abs(scoreline_expected(LAM, MU, -0.05).sum() - len(x)) < 1e-6


def test_analytic_gradient_matches_finite_differences():
    r = np.random.default_rng(1)
    h, a, la, lb = _season(rng=r)
    lam, mu = np.exp(la[h] + lb[a] + 0.2), np.exp(la[a] + lb[h])
    x, y = simulate_dc_scores(lam, mu, -0.08, r)
    d = Design(list(range(20)), h, a, x, y, np.zeros(20, int))
    f = fit_poisson(d)
    theta = np.concatenate([f.theta + 0.05 * r.standard_normal(len(f.theta)), [-0.06]])
    assert gradient_check(d, theta, dc_negloglik_and_grad) < 1e-4


def test_rho_recovered_on_simulated_season_within_se():
    """Planted rho = -0.10 on 4 seasons' worth of a 20-team fixture list; |rho_hat - rho| < 2.5 bootstrap SE."""
    r = np.random.default_rng(115)
    h, a, la, lb = _season(rng=r)
    h4, a4 = np.tile(h, 4), np.tile(a, 4)
    lam, mu = np.exp(la[h4] + lb[a4] + np.log(1.25)), np.exp(la[a4] + lb[h4])
    x, y = simulate_dc_scores(lam, mu, -0.10, r)
    d = Design(list(range(20)), h4, a4, x, y, np.zeros(20, int))
    fd = fit_dixon_coles(d)
    assert fd.converged and fd.admissible[0] < fd.rho < fd.admissible[1]
    b = bootstrap_rho(d, fd, 60, r, spec=Spec(d))
    assert abs(fd.rho - (-0.10)) < 2.5 * b["se"], (fd.rho, b["se"])
    assert abs(fd.gamma - 1.25) < 0.1
