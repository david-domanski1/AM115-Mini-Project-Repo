"""Data cleaning: date parsing, implied probabilities, and the 380-match / 20-team invariants."""
import os

import numpy as np
import pandas as pd
import pytest

from src import data as D


def test_implied_probabilities_sum_to_one_and_overround():
    pH, pD, pA, over = D.implied_probabilities([2.0, 1.5], [3.4, 4.0], [3.6, 7.0])
    assert np.allclose(pH + pD + pA, 1.0)
    assert np.allclose(over, [1 / 2 + 1 / 3.4 + 1 / 3.6 - 1, 1 / 1.5 + 1 / 4 + 1 / 7 - 1])
    assert np.allclose(pH, (1 / np.array([2.0, 1.5])) / (over + 1))


def test_implied_probabilities_propagate_nan():
    pH, pD, pA, over = D.implied_probabilities([np.nan], [3.0], [3.0])
    assert np.isnan(pH).all() and np.isnan(over).all()


def test_parse_both_date_formats():
    d = D.parse_dates(pd.Series(["13/08/05", "08/08/2015", "01/02/2026"]))
    assert list(d.dt.year) == [2005, 2015, 2026]
    assert list(d.dt.month) == [8, 8, 2]
    assert list(d.dt.day) == [13, 8, 1]


def test_season_code():
    assert D.season_code("2015-16") == "1516"
    assert D.season_code("2005-06") == "0506"
    assert D.season_code("2099-00") == "9900"


@pytest.fixture(scope="module")
def matches():
    path = os.path.join(D.PROCESSED_DIR, "matches.csv")
    if os.path.exists(path):
        return D.load_matches()
    return D.build_matches(D.ALL_SEASONS, write=False, verbose=False)  # downloads (cached) raw files


def test_every_complete_season_has_380_matches_and_20_teams(matches):
    for season, g in matches.groupby("season"):
        if D.season_expected_complete(season):
            assert len(g) == 380, season
            assert len(set(g["home"]) | set(g["away"])) == 20, season
            assert (g.groupby("home").size() == 19).all(), season
            assert (g.groupby("away").size() == 19).all(), season


def test_no_missing_goals_and_outcome_consistent(matches):
    assert matches[["home_goals", "away_goals"]].notna().all().all()
    hg, ag = matches["home_goals"], matches["away_goals"]
    exp = np.where(hg > ag, "H", np.where(hg < ag, "A", "D"))
    assert (exp == matches["ftr"].values).all()


def test_implied_probability_columns_sum_to_one(matches):
    for key in D.BOOKMAKERS:
        p = matches[[f"p_{key}_H", f"p_{key}_D", f"p_{key}_A"]].dropna()
        assert np.allclose(p.sum(axis=1), 1.0)
        over = matches[f"overround_{key}"].dropna()
        assert (over > 0).all() and (over < 0.25).all()


def test_team_name_table_exists_and_covers_all_teams(matches):
    tab = pd.read_csv(D.TEAM_NAMES_CSV)
    assert set(matches["home"]) <= set(tab["team"])
    assert tab["raw_name"].is_unique
