from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer


def train_models(data_dir: str | None = None, model_dir: str | None = None, random_state: int = 42):
    root = Path(__file__).resolve().parents[1]
    model_path = Path(model_dir) if model_dir else root / "models"
    model_path.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(data_dir=data_dir)
    fe = FeatureEngineer(loader)

    X, y_home, y_away = fe.build_training_dataset()
    X = X.fillna(0.0)

    feature_columns = list(X.columns)

    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X,
        y_home,
        y_away,
        test_size=0.2,
        random_state=random_state,
    )

    home_model = RandomForestRegressor(
        n_estimators=600,
        random_state=random_state,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    away_model = RandomForestRegressor(
        n_estimators=600,
        random_state=random_state + 1,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    home_model.fit(X_train, y_home_train)
    away_model.fit(X_train, y_away_train)

    home_pred = np.clip(home_model.predict(X_test), 0, None)
    away_pred = np.clip(away_model.predict(X_test), 0, None)

    home_mae = mean_absolute_error(y_home_test, home_pred)
    away_mae = mean_absolute_error(y_away_test, away_pred)

    home_rmse = mean_squared_error(y_home_test, home_pred) ** 0.5
    away_rmse = mean_squared_error(y_away_test, away_pred) ** 0.5

    joblib.dump(home_model, model_path / "home_goal_model.joblib")
    joblib.dump(away_model, model_path / "away_goal_model.joblib")

    (model_path / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2))

    metrics = {
        "n_rows": int(len(X)),
        "n_features": int(len(feature_columns)),
        "home_mae": float(home_mae),
        "away_mae": float(away_mae),
        "home_rmse": float(home_rmse),
        "away_rmse": float(away_rmse),
        "feature_columns": feature_columns,
    }

    (model_path / "training_metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


def main():
    metrics = train_models()
    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()