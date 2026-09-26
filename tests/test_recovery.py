"""The fitter recovers planted parameters, and the analytic gradient is correct."""
import numpy as np
from scipy.optimize import approx_fprime

from src.model import Design, Spec, fit_poisson, negloglik_and_grad
from src.simulate import recovery_study, simulate_scores

N = 20


def round_robin(n=N):
    home, away = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                home.append(i)
                away.append(j)
    return np.array(home), np.array(away)


def make_design(x=None, y=None):
    h, a = round_robin()
    x = np.zeros(len(h), int) if x is None else x
    y = np.zeros(len(h), int) if y is None else y
    return Design(list(range(N)), h, a, x, y, np.zeros(N, int))


def test_fixture_list_is_380_matches():
    d = make_design()
    assert d.n_matches == 380 and d.n_teams == 20


def test_analytic_gradient_matches_finite_differences():
    rng = np.random.default_rng(115)
    d = make_design(rng.poisson(1.5, 380), rng.poisson(1.2, 380))
    for constraint in ("alpha", "both"):
        spec = Spec(d, constraint)
        theta = rng.normal(0, 0.3, spec.n_params)
        _, g = negloglik_and_grad(theta, d, spec)
        num = approx_fprime(theta, lambda t: negloglik_and_grad(t, d, spec)[0], 1e-6)
        assert np.abs(g - num).max() < 1e-3 * max(1.0, np.abs(g).max())


def test_free_parameter_count():
    d = make_design()
    assert Spec(d, "alpha").n_params == 40
    assert Spec(d, "both").n_params == 39


def test_equal_strengths_recover_planted_gamma():
    d = make_design()
    rec = recovery_study(d, np.zeros(N), np.zeros(N), np.log(1.3), n_sims=40, seed=115)
    assert rec["n_converged"] == 40
    # mean of gamma-hat within 3 standard errors of the planted value
    se = rec["gamma_sd"] / np.sqrt(rec["n_sims"])
    assert abs(rec["gamma_bias"]) < 3 * se + 1e-3
    assert rec["gamma_rmse"] < 0.12


def test_random_strengths_recovered():
    rng = np.random.default_rng(7)
    la = rng.normal(0, 0.3, N)
    la -= la.mean()
    lb = rng.normal(np.log(1.3), 0.25, N)
    d = make_design()
    rec = recovery_study(d, la, lb, np.log(1.25), n_sims=40, seed=115)
    assert rec["log_alpha_rmse"] < 0.2
    assert rec["log_beta_rmse"] < 0.2
    assert abs(rec["gamma_bias"]) < 0.03
    assert np.exp(rec["fitted_log_alpha"]).prod(axis=1) == __import__("pytest").approx(1.0, abs=1e-8)


def test_mle_reproduces_goal_means():
    rng = np.random.default_rng(3)
    d = make_design()
    x, y = simulate_scores(np.zeros(N), np.full(N, np.log(1.2)), np.log(1.3), d.home, d.away, rng)
    d = make_design(x, y)
    f = fit_poisson(d)
    from src.model import rates
    lam, mu = rates(f.log_alpha, f.log_beta, f.log_gamma, d.home, d.away)
    assert abs(lam.mean() - x.mean()) < 1e-5
    assert abs(mu.mean() - y.mean()) < 1e-5
