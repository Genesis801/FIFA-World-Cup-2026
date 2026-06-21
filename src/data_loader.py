from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.utils import normalize_columns


class DataLoader:
    def __init__(self, data_dir: Optional[str] = None):
        root = Path(__file__).resolve().parents[1]
        self.data_dir = Path(data_dir) if data_dir else root / "data"
        self._cache: Dict[str, pd.DataFrame] = {}

    def _read_csv(self, filename: str, required: bool = True) -> Optional[pd.DataFrame]:
        path = self.data_dir / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Missing file: {path}")
            return None

        df = pd.read_csv(path)
        df = normalize_columns(df)
        return df

    def _load(self, key: str, filename: str, required: bool = True) -> Optional[pd.DataFrame]:
        if key not in self._cache:
            self._cache[key] = self._read_csv(filename, required=required)
        return self._cache[key]

    @property
    def tournaments(self) -> pd.DataFrame:
        return self._load("tournaments", "wc_tournaments.csv", required=True)

    @property
    def team_appearances(self) -> pd.DataFrame:
        return self._load("team_appearances", "wc_team_appearances.csv", required=True)

    @property
    def matches(self) -> pd.DataFrame:
        return self._load("matches", "wc_matches_historical.csv", required=True)

    @property
    def alltime_stats(self) -> pd.DataFrame:
        return self._load("alltime_stats", "wc_team_alltime_stats.csv", required=True)

    @property
    def group_difficulty(self) -> pd.DataFrame:
        return self._load("group_difficulty", "wc_2026_group_difficulty.csv", required=True)

    @property
    def qualifying_summary(self) -> pd.DataFrame:
        return self._load("qualifying_summary", "wc_2026_qualifying_summary.csv", required=True)

    @property
    def coaches(self) -> pd.DataFrame:
        return self._load("coaches", "wc_coaches_2026.csv", required=True)

    @property
    def head_to_head(self) -> pd.DataFrame:
        return self._load("head_to_head", "wc_head_to_head.csv", required=True)

    @property
    def top_scorers(self) -> pd.DataFrame:
        return self._load("top_scorers", "wc_top_scorers_by_edition.csv", required=True)

    @property
    def teams_snapshot(self) -> pd.DataFrame:
        return self._load("teams_snapshot", "wc_2026_teams_snapshot.csv", required=True)

    @property
    def prediction_features(self) -> pd.DataFrame:
        return self._load("prediction_features", "wc_prediction_features_2026.csv", required=True)

    def list_teams(self) -> list[str]:
        candidates = []

        for df_name in ["teams_snapshot", "prediction_features", "group_difficulty", "qualifying_summary", "alltime_stats"]:
            df = getattr(self, df_name, None)
            if df is not None and "team" in df.columns:
                candidates.extend(df["team"].dropna().astype(str).tolist())

        unique = sorted({c.strip() for c in candidates if str(c).strip()})
        return unique

    def get_tournament_host_map(self) -> Dict[int, str]:
        df = self.tournaments
        if "wc_year" not in df.columns or "host" not in df.columns:
            return {}
        out = {}
        for _, row in df.iterrows():
            try:
                out[int(row["wc_year"])] = str(row["host"]).strip()
            except Exception:
                continue
        return out