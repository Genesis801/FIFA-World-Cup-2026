from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.utils import (
    best_finish_to_score,
    coerce_numeric,
    clip_value,
    era_to_score,
    normalize_team_name,
    safe_div,
    stage_to_score,
)


class FeatureEngineer:
    def __init__(self, loader: DataLoader):
        self.loader = loader

    def _team_row(self, df: pd.DataFrame, team: str) -> Optional[pd.Series]:
        if df is None or "team" not in df.columns or not team:
            return None
        target = normalize_team_name(team)
        temp = df.copy()
        temp["_team_norm"] = temp["team"].astype(str).map(normalize_team_name)
        match = temp[temp["_team_norm"] == target]
        if match.empty:
            return None
        return match.iloc[0]

    def _series_value(self, row: Optional[pd.Series], col: str, default=np.nan):
        if row is None:
            return default
        if col not in row.index:
            return default
        return row.get(col, default)

    def _numeric(self, row: Optional[pd.Series], col: str, default: float = 0.0) -> float:
        return coerce_numeric(self._series_value(row, col, default), default=default)

    def team_profile(self, team: str) -> Dict[str, object]:
        pred = self._team_row(self.loader.prediction_features, team)
        qual = self._team_row(self.loader.qualifying_summary, team)
        group = self._team_row(self.loader.group_difficulty, team)
        coach = self._team_row(self.loader.coaches, team)
        snap = self._team_row(self.loader.teams_snapshot, team)
        alltime = self._team_row(self.loader.alltime_stats, team)

        coach_since = self._numeric(coach, "coach_since", default=np.nan)
        coach_experience_years = int(max(0, 2026 - coach_since)) if not pd.isna(coach_since) else np.nan

        profile = {
            "team": team,
            "confederation": self._series_value(snap, "confederation", self._series_value(pred, "confederation", "")),
            "is_debut": self._series_value(snap, "is_debut", False),
            "best_wc_finish": self._series_value(snap, "best_wc_finish", self._series_value(alltime, "best_finish", "")),
            "current_elo": self._numeric(pred, "elo_rating_2026", self._numeric(group, "elo_rating", self._numeric(alltime, "elo_peak_approx", 1500.0))),
            "current_fifa_rank": self._numeric(pred, "fifa_rank_apr2026", self._numeric(group, "fifa_rank", np.nan)),
            "market_value_eur_m": self._numeric(pred, "squad_market_value_eur_m", 0.0),
            "recent_form_pts_last10": self._numeric(pred, "recent_form_pts_last10", 0.0),
            "prediction_win_probability_pct": self._numeric(pred, "prediction_win_probability_pct", np.nan),
            "group_difficulty_index": self._numeric(group, "difficulty_index", 0.0),
            "group_avg_elo_excl_self": self._numeric(group, "group_avg_elo_excl_self", np.nan),
            "group_avg_fifa_rank_excl_self": self._numeric(group, "group_avg_fifa_rank_excl_self", np.nan),
            "group_max_elo_opponent": self._numeric(group, "group_max_elo_opponent", np.nan),
            "group_min_elo_opponent": self._numeric(group, "group_min_elo_opponent", np.nan),
            "expected_pts_group_stage": self._numeric(group, "expected_pts_group_stage", np.nan),
            "qualification_probability_pct": self._numeric(group, "qualification_probability_pct", np.nan),
            "qualifying_gf": self._numeric(qual, "qualifying_gf", 0.0),
            "qualifying_ga": self._numeric(qual, "qualifying_ga", 0.0),
            "qualifying_gd": self._numeric(qual, "qualifying_gd", 0.0),
            "qualifying_pts": self._numeric(qual, "qualifying_pts", 0.0),
            "qualifying_win_rate": self._numeric(qual, "qualifying_win_rate", 0.0),
            "qualifying_wins": self._numeric(qual, "qualifying_wins", 0.0),
            "qualifying_draws": self._numeric(qual, "qualifying_draws", 0.0),
            "qualifying_losses": self._numeric(qual, "qualifying_losses", 0.0),
            "qualification_route": self._series_value(qual, "qualification_route", ""),
            "coach_name": self._series_value(coach, "coach_name", ""),
            "coach_nationality": self._series_value(coach, "nationality", ""),
            "coach_age_at_wc2026": self._numeric(coach, "age_at_wc2026", np.nan),
            "coach_since": coach_since,
            "coach_experience_years": coach_experience_years,
            "wc_appearances_as_coach": self._numeric(coach, "wc_appearances_as_coach", 0.0),
            "wc_best_finish_as_coach": self._series_value(coach, "wc_best_finish_as_coach", ""),
            "coach_style": self._series_value(coach, "coaching_style", ""),
            "coach_achievement": self._series_value(coach, "notable_achievement", ""),
            "total_wc_appearances": self._numeric(alltime, "total_wc_appearances", 0.0),
            "win_rate": self._numeric(alltime, "win_rate", 0.0),
            "best_finish_score": best_finish_to_score(self._series_value(alltime, "best_finish", "")),
            "titles": self._numeric(alltime, "titles", 0.0),
            "elo_peak_approx": self._numeric(alltime, "elo_peak_approx", self._numeric(pred, "elo_rating_2026", 0.0)),
        }

        return profile

    def head_to_head_profile(self, team_a: str, team_b: str) -> Dict[str, float]:
        df = self.loader.head_to_head.copy()
        if df is None or df.empty:
            return {
                "h2h_total_matches": 0.0,
                "h2h_wins_a": 0.0,
                "h2h_wins_b": 0.0,
                "h2h_draws": 0.0,
                "h2h_goal_diff_a": 0.0,
                "h2h_last_wc_meeting_year": 0.0,
                "h2h_last_wc_meeting_stage_score": 0.0,
                "h2h_goal_share_a": 0.0,
                "h2h_win_rate_a": 0.0,
                "h2h_win_rate_b": 0.0,
            }

        df["_a"] = df["team_a"].astype(str).map(normalize_team_name)
        df["_b"] = df["team_b"].astype(str).map(normalize_team_name)
        a = normalize_team_name(team_a)
        b = normalize_team_name(team_b)

        direct = df[(df["_a"] == a) & (df["_b"] == b)]
        reverse = df[(df["_a"] == b) & (df["_b"] == a)]

        row = None
        flipped = False
        if not direct.empty:
            row = direct.iloc[0]
        elif not reverse.empty:
            row = reverse.iloc[0]
            flipped = True

        if row is None:
            return {
                "h2h_total_matches": 0.0,
                "h2h_wins_a": 0.0,
                "h2h_wins_b": 0.0,
                "h2h_draws": 0.0,
                "h2h_goal_diff_a": 0.0,
                "h2h_last_wc_meeting_year": 0.0,
                "h2h_last_wc_meeting_stage_score": 0.0,
                "h2h_goal_share_a": 0.0,
                "h2h_win_rate_a": 0.0,
                "h2h_win_rate_b": 0.0,
            }

        total = self._numeric(row, "total_wc_matches", 0.0)
        wins_a = self._numeric(row, "team_a_wins", 0.0)
        wins_b = self._numeric(row, "team_b_wins", 0.0)
        draws = self._numeric(row, "draws", 0.0)
        gd_a = self._numeric(row, "goal_difference_a", 0.0)
        last_year = self._numeric(row, "last_wc_meeting_year", 0.0)
        last_stage = stage_to_score(self._series_value(row, "last_wc_meeting_stage", ""))
        goals_a = self._numeric(row, "team_a_goals", 0.0)
        goals_b = self._numeric(row, "team_b_goals", 0.0)

        if flipped:
            wins_a, wins_b = wins_b, wins_a
            gd_a = -gd_a
            goals_a, goals_b = goals_b, goals_a

        goal_share_a = safe_div(goals_a, goals_a + goals_b, default=0.0)

        return {
            "h2h_total_matches": total,
            "h2h_wins_a": wins_a,
            "h2h_wins_b": wins_b,
            "h2h_draws": draws,
            "h2h_goal_diff_a": gd_a,
            "h2h_last_wc_meeting_year": last_year,
            "h2h_last_wc_meeting_stage_score": last_stage,
            "h2h_goal_share_a": goal_share_a,
            "h2h_win_rate_a": safe_div(wins_a, total, default=0.0),
            "h2h_win_rate_b": safe_div(wins_b, total, default=0.0),
        }

    def _alltime_row(self, team: str) -> pd.Series | None:
        return self._team_row(self.loader.alltime_stats, team)

    def _build_model_row(
        self,
        home_team: str,
        away_team: str,
        home_elo: float,
        away_elo: float,
        year: int,
        stage: str,
        host_team: Optional[str] = None,
    ) -> Dict[str, float]:
        home_all = self._alltime_row(home_team)
        away_all = self._alltime_row(away_team)

        home_titles = self._numeric(home_all, "titles", 0.0)
        away_titles = self._numeric(away_all, "titles", 0.0)

        home_apps = self._numeric(home_all, "total_wc_appearances", 0.0)
        away_apps = self._numeric(away_all, "total_wc_appearances", 0.0)

        home_wr = self._numeric(home_all, "win_rate", 0.0)
        away_wr = self._numeric(away_all, "win_rate", 0.0)

        home_best = best_finish_to_score(self._series_value(home_all, "best_finish", ""))
        away_best = best_finish_to_score(self._series_value(away_all, "best_finish", ""))

        home_peak = self._numeric(home_all, "elo_peak_approx", home_elo)
        away_peak = self._numeric(away_all, "elo_peak_approx", away_elo)

        host_norm = normalize_team_name(host_team) if host_team else ""

        host_is_home = 1.0 if host_norm and normalize_team_name(home_team) == host_norm else 0.0
        host_is_away = 1.0 if host_norm and normalize_team_name(away_team) == host_norm else 0.0

        if pd.isna(year):
            year = 2026

        year_norm = clip_value((float(year) - 1930.0) / (2026.0 - 1930.0), 0.0, 1.0)
        stage_code = stage_to_score(stage)
        era_code = era_to_score(stage)  # safe fallback; stage won't map, returns 0

        return {
            "home_elo": float(home_elo),
            "away_elo": float(away_elo),
            "elo_diff": float(home_elo) - float(away_elo),
            "elo_sum": float(home_elo) + float(away_elo),
            "home_titles": home_titles,
            "away_titles": away_titles,
            "titles_diff": home_titles - away_titles,
            "home_wc_appearances": home_apps,
            "away_wc_appearances": away_apps,
            "appearances_diff": home_apps - away_apps,
            "home_win_rate": home_wr,
            "away_win_rate": away_wr,
            "win_rate_diff": home_wr - away_wr,
            "home_best_finish_score": home_best,
            "away_best_finish_score": away_best,
            "best_finish_diff": home_best - away_best,
            "home_elo_peak": home_peak,
            "away_elo_peak": away_peak,
            "elo_peak_diff": home_peak - away_peak,
            "host_is_home": host_is_home,
            "host_is_away": host_is_away,
            "stage_code": float(stage_code),
            "era_code": float(era_code),
            "year_norm": float(year_norm),
        }

    def build_training_dataset(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        matches = self.loader.matches.copy()
        tournaments = self.loader.tournaments.copy()

        if "wc_year" not in matches.columns:
            raise ValueError("wc_matches_historical.csv must contain wc_year.")
        if "home_team" not in matches.columns or "away_team" not in matches.columns:
            raise ValueError("wc_matches_historical.csv must contain home_team and away_team.")
        if "home_goals" not in matches.columns or "away_goals" not in matches.columns:
            raise ValueError("wc_matches_historical.csv must contain home_goals and away_goals.")

        host_map = {}
        era_map = {}
        if "wc_year" in tournaments.columns:
            for _, row in tournaments.iterrows():
                try:
                    year = int(coerce_numeric(row.get("wc_year")))
                except Exception:
                    continue
                host_map[year] = str(row.get("host", "")).strip()
                era_map[year] = str(row.get("format_era", "")).strip()

        rows = []
        y_home = []
        y_away = []

        for _, m in matches.iterrows():
            try:
                year = int(coerce_numeric(m.get("wc_year")))
            except Exception:
                continue

            home_team = str(m.get("home_team", "")).strip()
            away_team = str(m.get("away_team", "")).strip()
            stage = str(m.get("stage", "")).strip()

            home_elo = coerce_numeric(m.get("home_pre_match_elo"), default=np.nan)
            away_elo = coerce_numeric(m.get("away_pre_match_elo"), default=np.nan)
            if pd.isna(home_elo):
                home_elo = 1500.0
            if pd.isna(away_elo):
                away_elo = 1500.0

            host_team = host_map.get(year, "")
            model_row = self._build_model_row(
                home_team=home_team,
                away_team=away_team,
                home_elo=home_elo,
                away_elo=away_elo,
                year=year,
                stage=stage,
                host_team=host_team,
            )
            model_row["era_code"] = float(era_to_score(era_map.get(year, "")))
            rows.append(model_row)
            y_home.append(coerce_numeric(m.get("home_goals"), default=0.0))
            y_away.append(coerce_numeric(m.get("away_goals"), default=0.0))

        X = pd.DataFrame(rows).fillna(0.0)
        y_home = pd.Series(y_home, name="home_goals")
        y_away = pd.Series(y_away, name="away_goals")

        return X, y_home, y_away

    def build_inference_features(
        self,
        home_team: str,
        away_team: str,
        stage: str = "Group Stage",
        host_team: Optional[str] = None,
        year: int = 2026,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]], Dict[str, float]]:
        home_profile = self.team_profile(home_team)
        away_profile = self.team_profile(away_team)

        home_elo = home_profile["current_elo"]
        away_elo = away_profile["current_elo"]

        model_row = self._build_model_row(
            home_team=home_team,
            away_team=away_team,
            home_elo=home_elo,
            away_elo=away_elo,
            year=year,
            stage=stage,
            host_team=host_team,
        )
        X = pd.DataFrame([model_row]).fillna(0.0)

        h2h = self.head_to_head_profile(home_team, away_team)

        support = {
            "home": home_profile,
            "away": away_profile,
            "h2h": h2h,
        }
        return X, support, model_row