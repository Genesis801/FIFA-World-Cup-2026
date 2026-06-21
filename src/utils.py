from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


STAGE_ORDER = {
    "group stage": 0,
    "round of 32": 1,
    "round of 16": 1,
    "quarter-finals": 2,
    "quarterfinals": 2,
    "quarter finals": 2,
    "semi-finals": 3,
    "semifinals": 3,
    "semi finals": 3,
    "third place": 3,
    "third-place": 3,
    "runner-up": 4,
    "runner up": 4,
    "final": 4,
    "champion": 5,
    "winner": 5,
}

ERA_ORDER = {
    "pre-war": 0,
    "early": 1,
    "classic": 2,
    "expansion": 3,
    "modern": 4,
    "48-team": 5,
    "48 team": 5,
}

BEST_FINISH_ORDER = {
    "did not qualify": 0,
    "didn't qualify": 0,
    "no appearance": 0,
    "group stage": 1,
    "round of 16": 2,
    "quarter-finals": 3,
    "quarterfinals": 3,
    "semi-finals": 4,
    "semifinals": 4,
    "runner-up": 5,
    "runner up": 5,
    "final": 5,
    "winner": 6,
    "champion": 6,
}

TEAM_ALIASES = {
    "usa": "usa",
    "united states": "usa",
    "united states of america": "usa",
    "south korea": "korea republic",
    "korea republic": "korea republic",
    "czechia": "czech republic",
    "czech republic": "czech republic",
    "ivory coast": "cote d'ivoire",
    "côte d'ivoire": "cote d'ivoire",
    "iran": "iran",
    "iran, islamic republic of": "iran",
    "england": "england",
    "wales": "wales",
    "scotland": "scotland",
    "united arab emirates": "uae",
    "uae": "uae",
    "republic of ireland": "ireland",
    "ireland": "ireland",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in out.columns
    ]
    return out


def normalize_team_name(name: Any) -> str:
    if name is None:
        return ""
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(".", "")
    return TEAM_ALIASES.get(s, s)


def coerce_numeric(value: Any, default: float = np.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, np.number)) and not pd.isna(value):
        return float(value)

    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "na"}:
        return default

    s = s.replace(",", "")
    match = re.search(r"-?\d+(\.\d+)?", s)
    if not match:
        return default
    try:
        return float(match.group(0))
    except ValueError:
        return default


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator in (0, 0.0, None) or pd.isna(denominator):
        return default
    return float(numerator) / float(denominator)


def stage_to_score(stage: Any) -> int:
    if stage is None or pd.isna(stage):
        return 0
    s = str(stage).strip().lower()
    return STAGE_ORDER.get(s, 0)


def era_to_score(era: Any) -> int:
    if era is None or pd.isna(era):
        return 0
    s = str(era).strip().lower()
    for key, val in ERA_ORDER.items():
        if key in s:
            return val
    return 0


def best_finish_to_score(value: Any) -> int:
    if value is None or pd.isna(value):
        return 0
    s = str(value).strip().lower()
    for key, val in BEST_FINISH_ORDER.items():
        if key in s:
            return val
    return 0


def winner_from_goals(home_goals: float, away_goals: float, home_team: str, away_team: str) -> str:
    if home_goals > away_goals:
        return home_team
    if away_goals > home_goals:
        return away_team
    return "Draw"


def confidence_from_probs(home_win_prob: float, draw_prob: float, away_win_prob: float) -> float:
    return round(max(home_win_prob, draw_prob, away_win_prob), 2)


def build_scoreline_label(home_goals: int, away_goals: int) -> str:
    return f"{home_goals}-{away_goals}"


def top_scorelines_from_counts(score_counts: Dict[Tuple[int, int], int], top_n: int = 5) -> List[Tuple[str, float]]:
    total = sum(score_counts.values()) or 1
    ranked = sorted(score_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(build_scoreline_label(h, a), round(count / total * 100, 2)) for (h, a), count in ranked]


def clip_value(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if not vals:
        return default
    return float(sum(vals)) / len(vals)


def pick_first_non_null(*values: Any, default: Any = None) -> Any:
    for v in values:
        if v is not None and not pd.isna(v) and str(v).strip() != "":
            return v
    return default


def build_team_stats_table(stats: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for k, v in stats.items():
        rows.append({"metric": k, "value": v})
    return pd.DataFrame(rows)