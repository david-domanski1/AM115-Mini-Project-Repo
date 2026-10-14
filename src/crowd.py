"""Crowd status by match date, and a Poisson fit with one home-advantage multiplier per crowd status.

Model (season-specific strengths, crowd-status-specific home advantage):

    X_m ~ Poisson(alpha_{s,h} * beta_{s,a} * gamma_{c(m)}),   Y_m ~ Poisson(alpha_{s,a} * beta_{s,h})

where s is the season of match m, h/a its home/away team and c(m) its crowd status
('full', 'closed', 'limited').  Identifiability: geometric-mean(alpha) = 1 within each
season, exactly as in ``src.model``; the gammas need no constraint.

Attendance is NOT in the football-data.co.uk files.  Crowd status is assigned from the
match date (and, for December 2020, the home club) using the public-health rules of the
time, listed below.  It is a flag for "fans were or were not allowed", not a head count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .data import load_raw_season
from .model import LOG_BOUNDS, Design, Spec, make_design

STATUSES = ("full", "closed", "limited")

# Premier League (E0).
#  * Closed doors: from the 2019-20 restart on 17 June 2020 to the end of 2020-21,
#    except the limited-attendance matches below.
#  * Limited (up to 2,000 fans): 2-15 December 2020, home clubs in tier-2 areas only
#    (London, Liverpool City Region, Brighton & Hove, Hampshire).  This club list is
#    from the tier allocation of 2 December 2020 and is NOT verified against match
#    attendance records; ``crowd_status(..., dec2020_by_club=False)`` flags every match
#    in the window instead, as a sensitivity check.  London moved to tier 3 on 16 Dec.
#  * Limited (up to 10,000 fans or 25% of capacity): 18-23 May 2021, the last two rounds,
#    all grounds.
E0_RESTART = pd.Timestamp("2020-06-17")
E0_DEC2020 = (pd.Timestamp("2020-12-02"), pd.Timestamp("2020-12-15"))
E0_MAY2021 = (pd.Timestamp("2021-05-18"), pd.Timestamp("2021-05-23"))
E0_TIER2_HOME_CLUBS_DEC2020 = ("Arsenal", "Chelsea", "Crystal Palace", "Fulham", "Tottenham", "West Ham",
                               "Liverpool", "Everton", "Brighton", "Southampton")
CLOSED_END = pd.Timestamp("2021-06-30")   # end of 2020-21 in all three leagues

# Bundesliga (D1) restarted 16 May 2020, La Liga (SP1) 11 June 2020.  Their partial
# crowds (Bundesliga Sep-Oct 2020; capacity caps in 2021-22; a few La Liga matches at
# the end of 2020-21) are NOT flagged separately: every match from the restart to the
# end of 2020-21 is 'closed', every other match is 'full' (meaning "fans allowed").
RESTART = {"E0": E0_RESTART, "D1": pd.Timestamp("2020-05-16"), "SP1": pd.Timestamp("2020-06-11")}
LEAGUE_NAMES = {"E0": "Premier League", "D1": "Bundesliga", "SP1": "La Liga"}
CROWD_SEASONS = ["2019-20", "2020-21", "2021-22"]


def crowd_status(df: pd.DataFrame, league: str = "E0", dec2020_by_club: bool = True) -> pd.Series:
    """'full' / 'closed' / 'limited' per match, from ``date`` (and ``home`` for Dec 2020 in E0)."""
    date = pd.to_datetime(df["date"])
    status = pd.Series("full", index=df.index, dtype=object)
    closed = (date >= RESTART[league]) & (date <= CLOSED_END)
    status[closed] = "closed"
    if league == "E0":
        dec = (date >= E0_DEC2020[0]) & (date <= E0_DEC2020[1])
        if dec2020_by_club:
            dec &= df["home"].isin(E0_TIER2_HOME_CLUBS_DEC2020)
        may = (date >= E0_MAY2021[0]) & (date <= E0_MAY2021[1])
        status[dec | may] = "limited"
    return status


def load_league(league: str, seasons=CROWD_SEASONS) -> pd.DataFrame:
    """Tidy match table for another league from the cached raw files (no download if cached).

    The canonical team name is the football-data short name (one spelling per club in a
    file, so no mapping table is needed for a within-league fit).
    """
    frames = []
    for s in seasons:
        r = load_raw_season(s, league=league)
        frames.append(pd.DataFrame({"season": s, "date": r["date"], "home": r["home_raw"], "away": r["away_raw"],
                                    "home_goals": r["home_goals"], "away_goals": r["away_goals"]}))
    df = pd.concat(frames, ignore_index=True).sort_values(["season", "date", "home"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- group fit
@dataclass
class GroupFit:
    teams: list
    groups: list                # group labels, index = gamma index
    log_alpha: np.ndarray
    log_beta: np.ndarray
    log_gamma: np.ndarray       # one per group
    theta: np.ndarray
    loglik: float
    converged: bool
    n_iter: int
    max_abs_theta: float
    proj_grad_norm: float

    @property
    def gamma(self):
        return dict(zip(self.groups, np.exp(self.log_gamma)))


class GroupSpec:
    """Parameter layout: free log-alpha (constraint per season), log-beta, one log-gamma per group."""

    def __init__(self, design: Design, n_groups: int):
        self.base = Spec(design, "alpha")
        self.K = n_groups
        self.na = self.base.n_free_a
        self.n = design.n_teams
        self.n_params = self.na + self.n + n_groups

    def unpack(self, theta):
        la = self.base._expand(theta[:self.na])
        lb = np.asarray(theta[self.na:self.na + self.n])
        return la, lb, np.asarray(theta[self.na + self.n:])


def group_negloglik_and_grad(theta, design: Design, spec: GroupSpec, gidx, weights=None):
    la, lb, lg = spec.unpack(theta)
    h, a = design.home, design.away
    lam = np.exp(la[h] + lb[a] + lg[gidx])
    mu = np.exp(la[a] + lb[h])
    w = np.ones_like(lam) if weights is None else weights
    x, y = design.x, design.y
    ll = np.sum(w * (x * np.log(lam) - lam + y * np.log(mu) - mu))
    rx, ry = w * (x - lam), w * (y - mu)
    n = design.n_teams
    da = np.bincount(h, rx, n) + np.bincount(a, ry, n)
    db = np.bincount(a, rx, n) + np.bincount(h, ry, n)
    dg = np.bincount(gidx, rx, spec.K)
    grad = np.concatenate([spec.base._reduce_grad(da), db, dg])
    return -ll, -grad


def projected_gradient_norm(theta, grad, bounds=LOG_BOUNDS):
    """Infinity norm of the box-projected gradient theta - clip(theta - grad, lo, hi)."""
    pg = theta - np.clip(theta - grad, bounds[0], bounds[1])
    return float(np.abs(pg).max())


def fit_groups(design: Design, gidx, labels, weights=None, x0=None, spec: GroupSpec | None = None) -> GroupFit:
    """MLE of the crowd-status model by L-BFGS-B with the analytic gradient."""
    gidx = np.asarray(gidx, int)
    spec = spec or GroupSpec(design, len(labels))
    w = np.ones(design.n_matches) if weights is None else np.asarray(weights, float)
    if x0 is None:
        x0 = np.zeros(spec.n_params)
        mx, my = np.average(design.x, weights=w), np.average(design.y, weights=w)
        x0[spec.na:spec.na + spec.n] = np.log(max(my, 1e-3))
        x0[spec.na + spec.n:] = np.log(max(mx, 1e-3) / max(my, 1e-3))
    res = minimize(group_negloglik_and_grad, x0, args=(design, spec, gidx, w), jac=True, method="L-BFGS-B",
                   bounds=[LOG_BOUNDS] * spec.n_params, options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-8})
    la, lb, lg = spec.unpack(res.x)
    lam = np.exp(la[design.home] + lb[design.away] + lg[gidx])
    mu = np.exp(la[design.away] + lb[design.home])
    full_ll = float(np.sum(w * (design.x * np.log(lam) - lam - gammaln(design.x + 1)
                                + design.y * np.log(mu) - mu - gammaln(design.y + 1))))
    _, g = group_negloglik_and_grad(res.x, design, spec, gidx, w)
    return GroupFit(list(design.teams), list(labels), la, lb, lg, res.x, full_ll, bool(res.success), int(res.nit),
                    float(np.abs(res.x).max()), projected_gradient_norm(res.x, g))


def crowd_design(df: pd.DataFrame, status: pd.Series, labels=STATUSES):
    """(Design with season-specific strengths, group index per match, labels present)."""
    present = [s for s in labels if (status == s).any()]
    d = make_design(df, season="season")
    gidx = status.map({s: i for i, s in enumerate(present)}).to_numpy(int)
    return d, gidx, present


def bootstrap_groups(design: Design, gidx, labels, fit: GroupFit, weights_matrix) -> dict:
    """Refit for every row of ``weights_matrix`` (bootstrap multiplicities); warm-started at ``fit``."""
    spec = GroupSpec(design, len(labels))
    B = len(weights_matrix)
    lg = np.empty((B, len(labels)))
    conv = 0
    max_theta, max_pg = 0.0, 0.0
    for b in range(B):
        fb = fit_groups(design, gidx, labels, weights=weights_matrix[b], x0=fit.theta, spec=spec)
        lg[b] = fb.log_gamma
        conv += fb.converged
        max_theta, max_pg = max(max_theta, fb.max_abs_theta), max(max_pg, fb.proj_grad_norm)
    return {"log_gamma_samples": lg, "gamma_samples": np.exp(lg), "n_converged": int(conv),
            "max_abs_theta": max_theta, "max_proj_grad_norm": max_pg}
