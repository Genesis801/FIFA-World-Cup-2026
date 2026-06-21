from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from src.utils import confidence_from_probs, top_scorelines_from_counts, winner_from_goals


class MatchSimulator:
    def simulate_match(
        self,
        home_team: str,
        away_team: str,
        home_xg: float,
        away_xg: float,
        n_simulations: int = 10000,
        max_goals: int = 10,
        random_state: int = 42,
    ) -> Dict[str, object]:
        rng = np.random.default_rng(random_state)

        home_scores = rng.poisson(lam=max(home_xg, 0.1), size=n_simulations)
        away_scores = rng.poisson(lam=max(away_xg, 0.1), size=n_simulations)

        score_counter = Counter()
        home_wins = 0
        away_wins = 0
        draws = 0

        for hg, ag in zip(home_scores, away_scores):
            hg_i = int(min(hg, max_goals))
            ag_i = int(min(ag, max_goals))
            score_counter[(hg_i, ag_i)] += 1

            if hg_i > ag_i:
                home_wins += 1
            elif ag_i > hg_i:
                away_wins += 1
            else:
                draws += 1

        home_win_prob = round(home_wins / n_simulations * 100, 2)
        away_win_prob = round(away_wins / n_simulations * 100, 2)
        draw_prob = round(draws / n_simulations * 100, 2)

        top_scorelines = top_scorelines_from_counts(dict(score_counter), top_n=5)

        best_score = max(score_counter.items(), key=lambda x: x[1])[0]
        best_scoreline = f"{best_score[0]}-{best_score[1]}"
        winner = winner_from_goals(best_score[0], best_score[1], home_team, away_team)
        confidence = confidence_from_probs(home_win_prob, draw_prob, away_win_prob)

        return {
            "home_win_prob_pct": home_win_prob,
            "draw_prob_pct": draw_prob,
            "away_win_prob_pct": away_win_prob,
            "predicted_scoreline": best_scoreline,
            "predicted_winner": winner,
            "confidence_pct": confidence,
            "top_scorelines": top_scorelines,
            "score_counts": dict(score_counter),
        }