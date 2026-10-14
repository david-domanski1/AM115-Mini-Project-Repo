"""Small checks of src/rolling.py (no full rolling run)."""
import numpy as np
import pandas as pd
import pytest

from src import rolling as R


def test_decay_weights_half_life_and_no_decay():
    w = R.decay_weights([0, 365, 730], 365)
    assert np.allclose(w, [1.0, 0.5, 0.25])
    assert np.allclose(R.decay_weights([0, 1000], None), 1.0)


def test_matchday_windows_split_fri_mon_and_tue_thu():
    dates = pd.to_datetime(["2024-08-16", "2024-08-17", "2024-08-19",   # Fri, Sat, Mon
                            "2024-08-20", "2024-08-22",                 # Tue, Thu
                            "2024-08-23"])                               # next Fri
    wid, start = R.matchday_windows(dates)
    assert wid[0] == wid[1] == wid[2] != wid[3] == wid[4] != wid[5]
    assert start[2] == pd.Timestamp("2024-08-16") and start[4] == pd.Timestamp("2024-08-20")
    assert all(s <= d for s, d in zip(start, dates))


def test_spell_labels_restart_after_gap():
    df = pd.DataFrame({"season": ["2005-06", "2006-07", "2008-09", "2008-09"],
                       "home": ["A", "A", "A", "B"], "away": ["B", "B", "B", "A"]})
    sp = R.spell_labels(df).set_index(["season", "team"])["spell"]
    assert sp[("2006-07", "A")] == "A|2005-06"
    assert sp[("2008-09", "A")] == "A|2008-09"       # gap in 2007-08 starts a new spell
    assert sp[("2008-09", "B")] == "B|2008-09"


def _toy_layout():
    """Two seasons; team C is relegated after season 1 and D promoted for season 2."""
    rows = []
    rng = np.random.default_rng(0)
    teams1, teams2 = ["A", "B", "C"], ["A", "B", "D"]
    day = pd.Timestamp("2010-08-06")    # a Friday
    for season, teams in (("2010-11", teams1), ("2011-12", teams2)):
        for r in range(6):
            for h in teams:
                for a in teams:
                    if h != a:
                        rows.append((season, day, h, a, rng.poisson(1.5), rng.poisson(1.1)))
            day += pd.Timedelta(days=7)
        day += pd.Timedelta(days=70)
    df = pd.DataFrame(rows, columns=["season", "date", "home", "away", "home_goals", "away_goals"])
    df["outcome"] = np.where(df.home_goals > df.away_goals, "H", np.where(df.home_goals == df.away_goals, "D", "A"))
    return R.make_layout(df)


def test_promoted_rules_targets():
    L = _toy_layout()
    n_e = len(L.spells)
    theta = np.zeros(2 * n_e + 1)
    i = {e: k for k, e in enumerate(L.spells)}
    theta[i["A|2010-11"]], theta[i["B|2010-11"]], theta[i["C|2010-11"]] = 0.3, 0.0, -0.3
    theta[n_e + i["C|2010-11"]] = 0.2
    _, t_rel, prom, rel = R.promoted_targets(theta, L, "2011-12", "relegated")
    assert prom == ["D|2011-12"] and rel == ["C|2010-11"]
    assert np.allclose(t_rel[0], [-0.3, 0.2])
    _, t_avg, _, _ = R.promoted_targets(theta, L, "2011-12", "league_avg")
    assert np.allclose(t_avg[0], [0.0, 0.2 / 3])
    _, t_sh, _, _ = R.promoted_targets(theta, L, "2011-12", "shrink50")
    assert np.allclose(t_sh[0], 0.5 * (t_rel[0] + t_avg[0]))


def test_rolling_forecast_uses_only_past_and_promoted_target_first():
    L = _toy_layout()
    r = R.rolling_forecast(L, "2011-12", half_life=None, rule="relegated")
    assert np.allclose(r["p"].sum(axis=1), 1.0)
    # first window: D has played nothing, so its forecast uses exactly the relegated-team target
    first = r["idx"][L.wid[r["idx"]] == L.wid[r["idx"]].min()]
    assert len(first) == 6 and r["n_fail"] == 0
    # changing a future result must not change an earlier forecast
    L2 = _toy_layout()
    last = r["idx"][-1]
    L2.x[last] += 5
    r2 = R.rolling_forecast(L2, "2011-12", half_life=None, rule="relegated")
    assert np.allclose(r["p"], r2["p"])


def test_scoring_toy_case():
    p = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    oi = np.array([0, 1])
    assert np.allclose(R.per_match_logloss(p, oi), [-np.log(0.5), -np.log(0.3)])
    assert np.allclose(R.per_match_brier(p, oi), [0.25 + 0.09 + 0.04, 0.04 + 0.49 + 0.25])
    out = R.paired_bootstrap({"a": [1.0, 1.0, 1.0], "b": [0.0, 0.0, 0.0]}, np.random.default_rng(1), 50, pairs=[("a", "b")])
    assert out["a-b"]["est"] == 1.0 and out["a-b"]["ci"] == [1.0, 1.0]


def test_calibration_binning():
    rng = np.random.default_rng(3)
    q = rng.dirichlet([4, 2, 3], size=1000)
    oi = np.array([rng.choice(3, p=pi) for pi in q])
    tab = R.reliability_table(q, oi, n_bins=5)
    assert len(tab) == 15 and (tab.groupby("outcome")["n"].sum() == 1000).all()
    h = tab[tab.outcome == "H"]
    assert (np.diff(h["mean_pred"].to_numpy()) > 0).all()
    citl = R.calibration_in_the_large(q, oi)
    assert abs(sum(citl.values())) < 1e-12          # probabilities and indicators both sum to 1
    lo, hi = R.wilson_ci(5, 10)
    assert lo < 0.5 < hi


def test_flat_stake_profit():
    p = np.array([[0.6, 0.2, 0.2]])
    odds = np.array([[2.0, 3.0, 6.0]])      # EV: +0.2, -0.4, +0.2
    prof, nb = R.flat_stake_profit(p, odds, np.array([0]), margin=0.0)
    assert nb[0] == 2 and prof[0] == pytest.approx(1.0 - 1.0)
    prof, nb = R.flat_stake_profit(p, odds, np.array([0]), margin=0.25)
    assert nb[0] == 0 and prof[0] == 0.0


def test_stratified_multiplicities_preserve_season_sizes():
    s = np.repeat([0, 1, 2], [5, 7, 4])
    m = R.stratified_multiplicities(s, np.random.default_rng(2), upto=1)
    assert m[s == 0].sum() == 5 and m[s == 1].sum() == 7 and m[s == 2].sum() == 0
