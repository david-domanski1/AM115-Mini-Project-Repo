"""Download, cache, clean and tidy football-data.co.uk Premier League CSVs.

Public entry point: ``build_matches(seasons)`` returns one tidy DataFrame with one
row per match, canonical team names, parsed dates and margin-free implied
probabilities for Bet365, Pinnacle (pre-match and closing) and the market average.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
TEAM_NAMES_CSV = os.path.join(ROOT, "data", "team_names.csv")
URL_PATTERN = "https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"
DEFAULT_LEAGUE = "E0"   # Premier League; D1 = Bundesliga, SP1 = La Liga (same file layout)

CORE_SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2015, 2026)]      # 2015-16 .. 2025-26
EXTENDED_SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2005, 2015)]  # 2005-06 .. 2014-15
ALL_SEASONS = EXTENDED_SEASONS + CORE_SEASONS

# Bookmaker key -> (home, draw, away) odds columns.  The market-average columns were
# renamed from BbAv* to Avg* in 2019-20, so that key lists both alternatives.
BOOKMAKERS = {
    "B365": [("B365H", "B365D", "B365A")],
    "PS": [("PSH", "PSD", "PSA")],
    "PSC": [("PSCH", "PSCD", "PSCA")],
    "Avg": [("BbAvH", "BbAvD", "BbAvA"), ("AvgH", "AvgD", "AvgA")],
}
# football-data.co.uk's notes file says the non-closing odds are "pre-closing" odds
# collected on Friday afternoons for weekend games and Tuesday afternoons for midweek
# games, i.e. 1-3 days before kick-off.  They are not opening odds.
BOOKMAKER_LABELS = {
    "B365": "Bet365 (pre-match, Fri/Tue collection)",
    "PS": "Pinnacle (pre-match, Fri/Tue collection)",
    "PSC": "Pinnacle (closing)",
    "Avg": "Market average (pre-match, Fri/Tue collection)",
}

# Full club names for display.  football-data.co.uk uses the same short name for a
# club in every file we downloaded (checked programmatically in team_name_table), so
# the canonical analysis name is the short name itself.
FULL_NAMES = {
    "Man City": "Manchester City", "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest", "QPR": "Queens Park Rangers",
    "West Brom": "West Bromwich Albion", "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers", "Tottenham": "Tottenham Hotspur",
    "Newcastle": "Newcastle United", "Leeds": "Leeds United", "Leicester": "Leicester City",
    "Brighton": "Brighton & Hove Albion", "Sheffield United": "Sheffield United",
    "Luton": "Luton Town", "Norwich": "Norwich City", "Stoke": "Stoke City",
    "Swansea": "Swansea City", "Hull": "Hull City", "Cardiff": "Cardiff City",
    "Huddersfield": "Huddersfield Town", "Ipswich": "Ipswich Town", "Bolton": "Bolton Wanderers",
    "Blackburn": "Blackburn Rovers", "Wigan": "Wigan Athletic", "Birmingham": "Birmingham City",
    "Charlton": "Charlton Athletic", "Derby": "Derby County", "Reading": "Reading",
    "Portsmouth": "Portsmouth", "Sunderland": "Sunderland", "Middlesbrough": "Middlesbrough",
    "Bournemouth": "AFC Bournemouth", "Blackpool": "Blackpool", "Burnley": "Burnley",
    "Watford": "Watford", "Fulham": "Fulham", "Everton": "Everton", "Chelsea": "Chelsea",
    "Arsenal": "Arsenal", "Liverpool": "Liverpool", "Aston Villa": "Aston Villa",
    "Crystal Palace": "Crystal Palace", "Southampton": "Southampton", "Brentford": "Brentford",
}


def season_code(label: str) -> str:
    """'2015-16' -> '1516'."""
    start = int(label[:4])
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def season_end_year(label: str) -> int:
    return int(label[:4]) + 1


def season_expected_complete(label: str, today: dt.date | None = None) -> bool:
    """A season ending in year Y is complete once we are past 1 July Y."""
    today = today or dt.date.today()
    return today >= dt.date(season_end_year(label), 7, 1)


def raw_path(label: str, league: str = DEFAULT_LEAGUE) -> str:
    return os.path.join(RAW_DIR, f"{league}_{season_code(label)}.csv")


def download_season(label: str, force: bool = False, verbose: bool = True,
                    league: str = DEFAULT_LEAGUE) -> str:
    """Download one season's CSV for ``league`` into data/raw/ (cached; never re-downloaded if present)."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = raw_path(label, league)
    if os.path.exists(path) and not force and os.path.getsize(path) > 0:
        return path
    url = URL_PATTERN.format(code=season_code(label), league=league)
    if verbose:
        print(f"[data] downloading {label} from {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AM115 mini-project)"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} for {url}: season {label} is not available. "
                           "Refusing to invent data.") from e
    if len(content) < 1000:
        raise RuntimeError(f"Suspiciously small file ({len(content)} bytes) for {url}")
    with open(path, "wb") as fh:
        fh.write(content)
    return path


def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def parse_dates(s: pd.Series) -> pd.Series:
    """football-data.co.uk mixes dd/mm/yy and dd/mm/yyyy; parse both explicitly."""
    s = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    short = s.str.len() == 8
    out[short] = pd.to_datetime(s[short], format="%d/%m/%y", errors="coerce")
    out[~short] = pd.to_datetime(s[~short], format="%d/%m/%Y", errors="coerce")
    return out


def implied_probabilities(odds_h, odds_d, odds_a):
    """Proportional (multiplicative) margin removal.

    p_k = (1/o_k) / sum_j (1/o_j).  Returns (pH, pD, pA, overround) where
    overround = sum_j 1/o_j - 1.  NaN odds propagate to NaN.
    """
    inv = np.column_stack([1.0 / np.asarray(odds_h, float),
                           1.0 / np.asarray(odds_d, float),
                           1.0 / np.asarray(odds_a, float)])
    total = inv.sum(axis=1)
    p = inv / total[:, None]
    return p[:, 0], p[:, 1], p[:, 2], total - 1.0


def load_raw_season(label: str, league: str = DEFAULT_LEAGUE) -> pd.DataFrame:
    """Read one raw file and return the used columns with a season label."""
    df = _read_csv(download_season(label, league=league))
    df = df.dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"]).copy()
    out = pd.DataFrame({
        "season": label,
        "date": parse_dates(df["Date"]),
        "home_raw": df["HomeTeam"].str.strip(),
        "away_raw": df["AwayTeam"].str.strip(),
        "home_goals": df["FTHG"].astype(int),
        "away_goals": df["FTAG"].astype(int),
        "ftr": df["FTR"].str.strip(),
    })
    for key, alternatives in BOOKMAKERS.items():
        cols = next((c for c in alternatives if all(x in df.columns for x in c)), None)
        for suffix, col in zip("HDA", cols or (None, None, None)):
            out[f"{key}{suffix}"] = pd.to_numeric(df[col], errors="coerce") if col else np.nan
    if out["date"].isna().any():
        raise ValueError(f"{label}: unparseable dates: {df.loc[out['date'].isna(), 'Date'].tolist()[:5]}")
    return out


def team_name_table(raw_names, write: bool = True) -> pd.DataFrame:
    """Build the raw -> canonical team-name mapping and save it to data/team_names.csv.

    The canonical name is the source short name (no club had more than one spelling
    across the downloaded files); a full display name is provided alongside.
    """
    names = sorted(set(raw_names))
    table = pd.DataFrame({
        "raw_name": names,
        "team": names,
        "full_name": [FULL_NAMES.get(n, n) for n in names],
    })
    if write:
        os.makedirs(os.path.dirname(TEAM_NAMES_CSV), exist_ok=True)
        table.to_csv(TEAM_NAMES_CSV, index=False)
    return table


def validate_seasons(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Assert 380 matches / 20 teams for complete seasons; report the partial ones."""
    info = {}
    for season, g in df.groupby("season", sort=True):
        n_teams = len(set(g["home"]) | set(g["away"]))
        n = len(g)
        complete = season_expected_complete(season)
        if complete:
            assert n == 380, f"{season}: expected 380 matches, found {n}"
            assert n_teams == 20, f"{season}: expected 20 teams, found {n_teams}"
        else:
            print(f"[data] {season} is in progress: {n} matches so far, {n_teams} teams seen")
        info[season] = {"n_matches": int(n), "n_teams": int(n_teams), "complete": bool(complete),
                        "first_date": str(g["date"].min().date()), "last_date": str(g["date"].max().date())}
        if verbose:
            print(f"[data] {season}: {n} matches, {n_teams} teams, "
                  f"{g['date'].min().date()} .. {g['date'].max().date()}"
                  f"{'' if complete else '  (partial)'}")
    return info


def build_matches(seasons=None, write: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Download (cached), clean and combine seasons into one tidy table."""
    seasons = list(seasons or ALL_SEASONS)
    frames = [load_raw_season(s) for s in seasons]
    df = pd.concat(frames, ignore_index=True)
    mapping = team_name_table(pd.concat([df["home_raw"], df["away_raw"]]), write=write)
    lookup = dict(zip(mapping["raw_name"], mapping["team"]))
    df["home"] = df["home_raw"].map(lookup)
    df["away"] = df["away_raw"].map(lookup)
    assert not df["home"].isna().any() and not df["away"].isna().any()
    # derived outcome and implied probabilities
    result = np.where(df["home_goals"] > df["away_goals"], "H",
                      np.where(df["home_goals"] < df["away_goals"], "A", "D"))
    assert (result == df["ftr"].values).all(), "FTR column disagrees with goals"
    df["outcome"] = pd.Categorical(result, categories=["H", "D", "A"])
    for key in BOOKMAKERS:
        pH, pD, pA, over = implied_probabilities(df[f"{key}H"], df[f"{key}D"], df[f"{key}A"])
        df[f"p_{key}_H"], df[f"p_{key}_D"], df[f"p_{key}_A"] = pH, pD, pA
        df[f"overround_{key}"] = over
    df = df.sort_values(["season", "date", "home"]).reset_index(drop=True)
    df = df.drop(columns=["home_raw", "away_raw"])
    cols = ["season", "date", "home", "away", "home_goals", "away_goals", "ftr", "outcome"]
    df = df[cols + [c for c in df.columns if c not in cols]]
    validate_seasons(df, verbose=verbose)
    if write:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df.to_csv(os.path.join(PROCESSED_DIR, "matches.csv"), index=False)
    return df


def load_matches(seasons=None) -> pd.DataFrame:
    """Load data/processed/matches.csv, building it if necessary."""
    path = os.path.join(PROCESSED_DIR, "matches.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        df["outcome"] = pd.Categorical(df["outcome"], categories=["H", "D", "A"])
        if seasons is None or set(seasons) <= set(df["season"]):
            return df if seasons is None else df[df["season"].isin(seasons)].reset_index(drop=True)
    return build_matches(seasons)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "2023-24"
    print(_read_csv(download_season(which)).columns.tolist())
