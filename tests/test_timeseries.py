"""Workstream B checks: the goals-ratio identity, crowd flags on known dates, Cochran's Q and
DerSimonian-Laird on toy tables, block construction, and the crowd-status fit."""
import numpy as np
import pandas as pd
import pytest

from src.crowd import crowd_design, crowd_status, fit_groups
from src.model import Design, fit_poisson, home_advantage_goals, make_design
from src.simulate import simulate_scores
from src.timeseries import (cluster_weights, cochran_q, date_blocks, dersimonian_laird, goals_ratio, iid_weights,
                            lr_common_gamma, percentile_ci, season_z, stratified_iid_weights, tau_qprofile_ci)


def round_robin(n=20):
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gamma_hat_equals_goals_ratio_on_simulated_round_robin(seed):
    rng = np.random.default_rng(seed)
    h, a = round_robin()
    la = rng.normal(0, 0.3, 20)
    la -= la.mean()
    lb = rng.normal(0.2, 0.3, 20)
    x, y = simulate_scores(la, lb, np.log(1.25), h, a, rng)
    d = Design(list(range(20)), h, a, x, y, np.zeros(20, int))
    f = fit_poisson(d)
    assert abs(f.gamma - x.sum() / y.sum()) < 1e-6
    assert abs(home_advantage_goals(f, d) - (x.mean() - y.mean())) < 1e-6


def test_pooled_common_gamma_equals_pooled_goals_ratio():
    rng = np.random.default_rng(7)
    h, a = round_robin(6)
    frames = []
    for s, g in (("A", 1.1), ("B", 1.4), ("C", 1.2)):
        la = rng.normal(0, 0.3, 6)
        la -= la.mean()
        x, y = simulate_scores(la, rng.normal(0.3, 0.2, 6), np.log(g), h, a, rng)
        frames.append(pd.DataFrame({"season": s, "home": h, "away": a, "home_goals": x, "away_goals": y}))
    df = pd.concat(frames, ignore_index=True)
    f = fit_poisson(make_design(df, season="season"))
    assert abs(f.gamma - goals_ratio(df["home_goals"], df["away_goals"])) < 1e-6


def test_identity_fails_for_unbalanced_weights():
    """The identity needs a balanced schedule: a bootstrap reweighting generally breaks it (so refits are needed)."""
    rng = np.random.default_rng(3)
    h, a = round_robin()
    x, y = simulate_scores(np.zeros(20), rng.normal(0, 0.4, 20), np.log(1.3), h, a, rng)
    d = Design(list(range(20)), h, a, x, y, np.zeros(20, int))
    w = iid_weights(380, 1, rng)[0].astype(float)
    f = fit_poisson(d, weights=w)
    assert abs(f.gamma - np.sum(w * x) / np.sum(w * y)) > 1e-6


def test_crowd_flags_on_known_dates():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-03-09", "2020-06-17", "2020-12-05", "2020-12-05", "2020-12-16",
                                "2021-05-16", "2021-05-18", "2021-05-23", "2021-08-14"]),
        "home": ["Leicester", "Aston Villa", "Chelsea", "Burnley", "Arsenal", "Chelsea", "Man United", "Fulham", "Brentford"],
        "away": ["Aston Villa", "Sheffield United", "Leeds", "Everton", "Southampton", "Arsenal", "Fulham", "Newcastle", "Arsenal"],
    })
    st = crowd_status(df, "E0")
    assert list(st) == ["full", "closed", "limited", "closed", "closed", "closed", "limited", "limited", "full"]
    st2 = crowd_status(df, "E0", dec2020_by_club=False)
    assert st2.iloc[3] == "limited" and st2.iloc[4] == "closed"
    d1 = crowd_status(pd.DataFrame({"date": pd.to_datetime(["2020-05-15", "2020-05-16", "2021-05-22", "2021-08-13"]),
                                    "home": ["x"] * 4}), "D1")
    assert list(d1) == ["full", "closed", "closed", "full"]


def test_cochran_q_toy_table():
    y = np.array([0.0, 0.2, 0.4])
    se = np.array([0.1, 0.1, 0.1])
    q = cochran_q(y, se)
    # equal weights: mean 0.2, Q = sum((y - 0.2)^2) / 0.01 = 8
    assert q["fe_mean"] == pytest.approx(0.2)
    assert q["Q"] == pytest.approx(8.0)
    assert q["df"] == 2
    assert q["p"] == pytest.approx(np.exp(-4.0))   # chi2 with 2 df: sf(8) = exp(-8/2)
    dl = dersimonian_laird(y, se)
    # C = sum w - sum w^2 / sum w = 300 - 100 = 200; tau^2 = (8 - 2) / 200 = 0.03
    assert dl["tau2"] == pytest.approx(0.03)
    lo, hi = dl["tau_ci_qprofile"]
    assert lo < dl["tau"] < hi


def test_homogeneous_table_gives_tau_zero_and_small_q():
    y = np.array([0.20, 0.21, 0.19, 0.20])
    se = np.full(4, 0.1)
    assert dersimonian_laird(y, se)["tau2"] == 0.0
    assert tau_qprofile_ci(y, se)[0] == 0.0
    z = season_z(0.20, 0.1, y, se)
    assert abs(z["z_common"]) < 0.1 and z["z_common"] == pytest.approx(z["z_random"])


def test_lr_common_gamma():
    r = lr_common_gamma([-10.0, -12.0, -11.0], -36.0, 3)
    assert r["LR"] == pytest.approx(6.0) and r["df"] == 2
    assert r["p"] == pytest.approx(np.exp(-3.0))


def test_date_blocks_are_38_consecutive_blocks_of_10():
    rng = np.random.default_rng(0)
    dates = pd.Series(pd.to_datetime("2024-08-16") + pd.to_timedelta(rng.integers(0, 280, 380), unit="D"))
    b = date_blocks(dates, 10)
    assert len(np.unique(b)) == 38 and np.all(np.bincount(b) == 10)
    order = np.argsort(dates.to_numpy(), kind="stable")
    assert np.all(np.diff(b[order]) >= 0)          # block ids increase with date
    for k in range(38):                             # every block spans a contiguous date range
        assert dates[b == k].max() <= dates[b == k + 1].min() if k < 37 else True


def test_cluster_and_iid_weights_sum_to_sample_size():
    rng = np.random.default_rng(1)
    blocks = np.repeat(np.arange(38), 10)
    W = cluster_weights(blocks, 50, rng)
    assert W.shape == (50, 380) and np.all(W.sum(axis=1) == 380)
    for w in W:                                     # all matches in a block share a weight
        assert np.all(w.reshape(38, 10) == w.reshape(38, 10)[:, :1])
    Wt = cluster_weights(np.repeat(np.arange(20), 19), 20, rng)
    assert np.all(Wt.sum(axis=1) == 380)
    Wi = iid_weights(380, 20, rng)
    assert np.all(Wi.sum(axis=1) == 380)
    Ws = stratified_iid_weights(np.repeat([0, 1, 2], [100, 50, 30]), 10, rng)
    assert np.all(Ws[:, :100].sum(axis=1) == 100) and np.all(Ws[:, 150:].sum(axis=1) == 30)


def test_iid_weights_reproduce_src_bootstrap_stream():
    """Same seed -> same replicates as src.bootstrap.bootstrap_fit (so 500-rep CIs match run_all.py)."""
    rng_a, rng_b = np.random.default_rng(115), np.random.default_rng(115)
    W = iid_weights(380, 3, rng_a)
    for b in range(3):
        draw = rng_b.choice(np.arange(380), size=380, replace=True)
        assert np.array_equal(W[b], np.bincount(draw, minlength=380))


def test_crowd_group_fit_recovers_planted_gammas():
    rng = np.random.default_rng(11)
    h, a = round_robin()
    frames = []
    for s in ("2019-20", "2020-21"):
        frames.append(pd.DataFrame({"season": s, "home": h, "away": a}))
    df = pd.concat(frames, ignore_index=True)
    status = pd.Series(np.where(np.arange(len(df)) % 2 == 0, "full", "closed"))
    d, gidx, present = crowd_design(df.assign(home_goals=0, away_goals=0), status, ("full", "closed"))
    la = rng.normal(0, 0.2, d.n_teams)
    for g in range(2):
        la[d.groups == g] -= la[d.groups == g].mean()
    lb = rng.normal(0.2, 0.2, d.n_teams)
    true = np.log([1.35, 1.0])
    xs, ys = [], []
    for _ in range(30):
        lam = np.exp(la[d.home] + lb[d.away] + true[gidx])
        mu = np.exp(la[d.away] + lb[d.home])
        xs.append(rng.poisson(lam)); ys.append(rng.poisson(mu))
    est = []
    for x, y in zip(xs, ys):
        dd = Design(d.teams, d.home, d.away, x, y, d.groups)
        f = fit_groups(dd, gidx, present)
        assert f.converged and f.proj_grad_norm < 1e-3
        est.append(f.log_gamma)
    est = np.array(est)
    assert np.all(np.abs(est.mean(axis=0) - true) < 3 * est.std(axis=0, ddof=1) / np.sqrt(30) + 0.01)


def test_trend_meta_regression_recovers_exact_line():
    from src.timeseries import trend_meta_regression
    t = np.arange(2005, 2025)
    y = 0.3 - 0.01 * (t - 2005)
    r = trend_meta_regression(y, np.full(len(t), 0.05), t)
    assert r["slope_log_gamma_per_season"] == pytest.approx(-0.01)
    assert r["residual_Q"] == pytest.approx(0.0, abs=1e-12)


def test_percentile_ci():
    assert percentile_ci(np.arange(1001)) == pytest.approx([25.0, 975.0])
