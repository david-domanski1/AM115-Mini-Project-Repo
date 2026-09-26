"""Score-grid outcome probabilities must agree with the Skellam closed form."""
import numpy as np
from scipy.stats import skellam

from src.model import outcome_probs, skellam_outcome_probs, skellam_pmf

rng = np.random.default_rng(115)
LAM = rng.uniform(0.2, 4.0, 300)
MU = rng.uniform(0.2, 4.0, 300)


def test_skellam_pmf_matches_scipy():
    for k in (-3, -1, 0, 1, 2, 5):
        assert np.allclose(skellam_pmf(k, LAM, MU), skellam.pmf(k, LAM, MU), atol=1e-14, rtol=0)


def test_grid_draw_probability_matches_skellam_to_1e10():
    grid = outcome_probs(LAM, MU, max_goals=40)
    sk = skellam_outcome_probs(LAM, MU)
    assert np.abs(grid[:, 1] - sk[:, 1]).max() < 1e-10
    assert np.abs(grid - sk).max() < 1e-10


def test_wide_grid_sums_to_one():
    p = outcome_probs(LAM, MU, max_goals=40)
    assert np.abs(p.sum(axis=1) - 1).max() < 1e-10


def test_default_grid_truncation_is_small_for_typical_rates():
    lam = rng.uniform(0.5, 3.0, 200)
    mu = rng.uniform(0.5, 3.0, 200)
    p10 = outcome_probs(lam, mu, max_goals=10)
    sk = skellam_outcome_probs(lam, mu)
    assert np.abs(p10 - sk).max() < 1e-3
    assert (p10.sum(axis=1) <= 1 + 1e-12).all()
