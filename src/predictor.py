from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.simulator import MatchSimulator
from src.utils import build_team_stats_table, clip_value, normalize_team_name


class MatchPredictor:
    def __init__(self, model_dir: Optional[str] = None, data_dir: Optional[str] = None):
        root = Path(__file__).resolve().parents[1]
        self.model_dir = Path(model_dir) if model_dir else root / "models"
        self.loader = DataLoader(data_dir=data_dir)
        self.feature_engineer = FeatureEngineer(self.loader)
        self.simulator = MatchSimulator()

        self.home_model = joblib.load(self.model_dir / "home_goal_model.joblib")
        self.away_model = joblib.load(self.model_dir / "away_goal_model.joblib")

        feature_path = self.model_dir / "feature_columns.json"
        if feature_path.exists():
            self.feature_columns = json.loads(feature_path.read_text())
        else:
            self.feature_columns = None

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.feature_columns is None:
            return X.fillna(0.0)

        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_columns]
        return X.fillna(0.0)

    def _normalize_0_1(self, value: float, min_value: float, max_value: float) -> float:
        if value is None or pd.isna(value):
            return 0.5
        if max_value == min_value:
            return 0.5
        return float(np.clip((value - min_value) / (max_value - min_value), 0.0, 1.0))

    def _inverse_rank_score(self, rank: float) -> float:
        if rank is None or pd.isna(rank) or rank <= 0:
            return 0.5
        return float(np.clip(1.0 - ((rank - 1.0) / 48.0), 0.0, 1.0))

    def _strength_score(self, profile: Dict[str, object]) -> float:
        elo = profile.get("current_elo", 1500.0)
        rank = profile.get("current_fifa_rank", np.nan)
        market = profile.get("market_value_eur_m", 0.0)
        form = profile.get("recent_form_pts_last10", 0.0)
        qual_gd = profile.get("qualifying_gd", 0.0)
        coach_exp = profile.get("coach_experience_years", 0.0)
        coach_apps = profile.get("wc_appearances_as_coach", 0.0)
        difficulty = profile.get("group_difficulty_index", 0.0)
        h2h_win_rate = profile.get("h2h_win_rate_a", 0.0)

        elo_score = self._normalize_0_1(float(elo), 1450.0, 2100.0)
        rank_score = self._inverse_rank_score(float(rank)) if not pd.isna(rank) else 0.5
        market_score = self._normalize_0_1(float(market), 0.0, 2500.0)
        form_score = self._normalize_0_1(float(form), 0.0, 30.0)
        qual_gd_score = self._normalize_0_1(float(qual_gd), -20.0, 40.0)
        coach_exp_score = self._normalize_0_1(float(coach_exp), 0.0, 20.0)
        coach_apps_score = self._normalize_0_1(float(coach_apps), 0.0, 6.0)
        difficulty_penalty = 1.0 - self._normalize_0_1(float(difficulty), 0.0, 1.0)
        h2h_score = self._normalize_0_1(float(h2h_win_rate), 0.0, 1.0)

        strength = (
            0.30 * elo_score
            + 0.16 * rank_score
            + 0.12 * market_score
            + 0.14 * form_score
            + 0.10 * qual_gd_score
            + 0.08 * coach_exp_score
            + 0.05 * coach_apps_score
            + 0.03 * difficulty_penalty
            + 0.02 * h2h_score
        )
        return float(np.clip(strength, 0.0, 1.0))

    def _support_adjustment(self, support: Dict[str, Dict[str, object]], stage: str, host_team: Optional[str]) -> float:
        home = support["home"]
        away = support["away"]

        home_strength = self._strength_score({
            **home,
            "h2h_win_rate_a": support.get("h2h", {}).get("h2h_win_rate_a", 0.0),
        })
        away_strength = self._strength_score({
            **away,
            "h2h_win_rate_a": support.get("h2h", {}).get("h2h_win_rate_b", 0.0),
        })

        diff = home_strength - away_strength
        adj = 0.14 * float(np.tanh(diff * 2.0))

        if host_team:
            host_norm = normalize_team_name(host_team)
            if normalize_team_name(home.get("team", "")) == host_norm:
                adj += 0.04
            elif normalize_team_name(away.get("team", "")) == host_norm:
                adj -= 0.04

        stage_lower = str(stage).strip().lower()
        if "final" in stage_lower or "semi" in stage_lower or "quarter" in stage_lower:
            adj *= 0.9

        return float(np.clip(adj, -0.20, 0.20))

    def _build_reasons(self, support: Dict[str, Dict[str, object]]) -> list[str]:
        home = support["home"]
        away = support["away"]
        h2h = support.get("h2h", {})

        reasons = []

        if home.get("current_elo", 0) > away.get("current_elo", 0):
            reasons.append("Higher current ELO rating.")
        elif away.get("current_elo", 0) > home.get("current_elo", 0):
            reasons.append("Opposition has the higher current ELO rating.")

        if home.get("qualifying_gd", 0) > away.get("qualifying_gd", 0):
            reasons.append("Better qualifying goal difference.")
        elif away.get("qualifying_gd", 0) > home.get("qualifying_gd", 0):
            reasons.append("Opponent had a better qualifying goal difference.")

        if home.get("recent_form_pts_last10", 0) > away.get("recent_form_pts_last10", 0):
            reasons.append("Stronger recent form.")
        elif away.get("recent_form_pts_last10", 0) > home.get("recent_form_pts_last10", 0):
            reasons.append("Opponent has stronger recent form.")

        if home.get("coach_experience_years", 0) > away.get("coach_experience_years", 0):
            reasons.append("More experienced coach.")
        elif away.get("coach_experience_years", 0) > home.get("coach_experience_years", 0):
            reasons.append("Opponent has the more experienced coach.")

        if h2h.get("h2h_total_matches", 0) > 0:
            if h2h.get("h2h_wins_a", 0) > h2h.get("h2h_wins_b", 0):
                reasons.append("Positive historical head-to-head record.")
            elif h2h.get("h2h_wins_b", 0) > h2h.get("h2h_wins_a", 0):
                reasons.append("Opponent has the better head-to-head record.")

        if not reasons:
            reasons.append("The prediction is driven mainly by the model's learned goal expectancy.")

        return reasons[:4]

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        stage: str = "Group Stage",
        host_team: Optional[str] = None,
        n_simulations: int = 10000,
        random_state: int = 42,
    ) -> Dict[str, object]:
        if normalize_team_name(home_team) == normalize_team_name(away_team):
            raise ValueError("Home team and away team must be different.")

        X, support, model_row = self.feature_engineer.build_inference_features(
            home_team=home_team,
            away_team=away_team,
            stage=stage,
            host_team=host_team,
            year=2026,
        )

        X_model = self._align_features(X)

        home_xg = float(self.home_model.predict(X_model)[0])
        away_xg = float(self.away_model.predict(X_model)[0])

        home_xg = max(home_xg, 0.10)
        away_xg = max(away_xg, 0.10)

        adj = self._support_adjustment(support, stage=stage, host_team=host_team)
        home_xg = max(0.10, home_xg * (1.0 + adj))
        away_xg = max(0.10, away_xg * (1.0 - adj))

        simulation = self.simulator.simulate_match(
            home_team=home_team,
            away_team=away_team,
            home_xg=home_xg,
            away_xg=away_xg,
            n_simulations=n_simulations,
            random_state=random_state,
        )

        home_stats = self._team_stats_for_display(support["home"])
        away_stats = self._team_stats_for_display(support["away"])
        reasons = self._build_reasons(support)

        report = {
            "home_team": home_team,
            "away_team": away_team,
            "stage": stage,
            "host_team": host_team,
            "home_xg": round(home_xg, 2),
            "away_xg": round(away_xg, 2),
            "predicted_winner": simulation["predicted_winner"],
            "predicted_scoreline": simulation["predicted_scoreline"],
            "confidence_pct": simulation["confidence_pct"],
            "home_win_prob_pct": simulation["home_win_prob_pct"],
            "draw_prob_pct": simulation["draw_prob_pct"],
            "away_win_prob_pct": simulation["away_win_prob_pct"],
            "top_scorelines": simulation["top_scorelines"],
            "reasons": reasons,
            "home_stats": home_stats,
            "away_stats": away_stats,
            "raw_model_features": model_row,
        }
        return report

    def _team_stats_for_display(self, profile: Dict[str, object]) -> Dict[str, object]:
        return {
            "Team": profile.get("team", ""),
            "Confederation": profile.get("confederation", ""),
            "Current ELO": round(float(profile.get("current_elo", 0.0)), 1),
            "FIFA Rank": profile.get("current_fifa_rank", ""),
            "Market Value (€m)": round(float(profile.get("market_value_eur_m", 0.0)), 1),
            "Recent Form (last 10)": profile.get("recent_form_pts_last10", ""),
            "Qualifying GD": profile.get("qualifying_gd", ""),
            "Qualifying Win Rate": profile.get("qualifying_win_rate", ""),
            "Coach": profile.get("coach_name", ""),
            "Coach Experience (yrs)": profile.get("coach_experience_years", ""),
            "WC Apps (coach)": profile.get("wc_appearances_as_coach", ""),
            "Titles": profile.get("titles", ""),
            "All-time Win Rate": profile.get("win_rate", ""),
            "Best Finish Score": profile.get("best_finish_score", ""),
            "ELO Peak": profile.get("elo_peak_approx", ""),
        }