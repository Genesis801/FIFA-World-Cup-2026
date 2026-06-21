from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import DataLoader
from src.predictor import MatchPredictor
from src.utils import build_team_stats_table, normalize_team_name


st.set_page_config(
    page_title="FIFA World Cup Match Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_loader() -> DataLoader:
    return DataLoader()


@st.cache_resource
def get_predictor() -> MatchPredictor:
    return MatchPredictor()


def stats_df(stats: dict) -> pd.DataFrame:
    return build_team_stats_table(stats)


def main():
    st.title("FIFA World Cup Match Predictor")
    st.caption("Regression-based expected goals model + Poisson simulation + supporting team stats.")

    loader = get_loader()
    predictor = get_predictor()

    teams = loader.list_teams()
    if not teams:
        st.error("No teams found. Check that your CSV files are inside the data/ folder.")
        return

    with st.sidebar:
        st.header("Match Context")
        stage = st.selectbox(
            "Stage",
            ["Group Stage", "Round of 16", "Quarter-finals", "Semi-finals", "Final"],
            index=0,
        )

        host_options = ["Neutral / no host"] + teams
        host_choice = st.selectbox("Host team", host_options, index=0)
        host_team = None if host_choice == "Neutral / no host" else host_choice

        st.divider()
        st.write("The model uses historical match regression outputs and adjusts them with 2026 supporting priors.")

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Team 1", teams, index=0)
    with col2:
        away_team = st.selectbox("Team 2", teams, index=1 if len(teams) > 1 else 0)

    if normalize_team_name(home_team) == normalize_team_name(away_team):
        st.warning("Please select two different teams.")
        return

    predict_clicked = st.button("Predict match", type="primary")

    if predict_clicked:
        try:
            report = predictor.predict_match(
                home_team=home_team,
                away_team=away_team,
                stage=stage,
                host_team=host_team,
                n_simulations=10000,
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        st.subheader("Prediction")

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Predicted winner", report["predicted_winner"])
        p2.metric("Predicted scoreline", report["predicted_scoreline"])
        p3.metric("Confidence", f'{report["confidence_pct"]:.2f}%')
        p4.metric("Expected goals", f'{report["home_xg"]} - {report["away_xg"]}')

        prob_col1, prob_col2, prob_col3 = st.columns(3)
        prob_col1.metric(f"{home_team} win", f'{report["home_win_prob_pct"]:.2f}%')
        prob_col2.metric("Draw", f'{report["draw_prob_pct"]:.2f}%')
        prob_col3.metric(f"{away_team} win", f'{report["away_win_prob_pct"]:.2f}%')

        st.subheader("Most likely scorelines")
        score_df = pd.DataFrame(report["top_scorelines"], columns=["scoreline", "probability_pct"])
        st.dataframe(score_df, use_container_width=True, hide_index=True)

        st.subheader("Why the model leaned this way")
        for reason in report["reasons"]:
            st.write(f"- {reason}")

        left, right = st.columns(2)
        with left:
            st.subheader(home_team)
            st.dataframe(stats_df(report["home_stats"]), use_container_width=True, hide_index=True)
        with right:
            st.subheader(away_team)
            st.dataframe(stats_df(report["away_stats"]), use_container_width=True, hide_index=True)

        st.subheader("Probability view")
        prob_plot = pd.DataFrame(
            {
                "outcome": ["Home win", "Draw", "Away win"],
                "probability": [
                    report["home_win_prob_pct"],
                    report["draw_prob_pct"],
                    report["away_win_prob_pct"],
                ],
            }
        ).set_index("outcome")
        st.bar_chart(prob_plot)

        with st.expander("Raw feature row used for prediction"):
            st.dataframe(pd.DataFrame([report["raw_model_features"]]), use_container_width=True)

    st.divider()
    st.info(
        "Tip: For a more realistic results, you can add a stage selector and host advantage. " \
        "The model uses historical match regression outputs and adjusts them with 2026 supporting priors. " \
        "#Cheers to the sprit of the beautiful game! D.R"
    )


if __name__ == "__main__":
    main()