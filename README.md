# FIFA World Cup Match Predictor

A modular machine learning application that predicts the outcome of a FIFA World Cup match between two teams and estimates a likely scoreline with confidence. The app combines historical World Cup match data, team strength indicators, qualification performance, coach information, head-to-head records, and 2026 tournament context.

---

## 1. Project Overview

This project is a **regression-based football match prediction system** built for World Cup analytics. It takes two teams as input and produces:

* Predicted winner
* Win / draw / loss probabilities
* Predicted scoreline
* Confidence score
* Supporting statistics for both teams
* Most likely scorelines from simulation

The system is packaged as a **Streamlit web application** and is organized into reusable Python modules so that the logic can be maintained, tested, and extended easily.

---

## 2. What the App Does

The application answers a practical football analytics question:

> Given Team A and Team B, who is more likely to win, what could the scoreline be, and why?

It does this by:

1. Loading multiple World Cup datasets.
2. Building a structured feature vector for the two selected teams.
3. Predicting expected goals for each side using regression models.
4. Simulating many possible match outcomes using Poisson distributions.
5. Converting the simulated outcomes into probabilities.
6. Showing supporting team statistics so the prediction is explainable.

The app is designed to feel like a lightweight production tool rather than a notebook demo.

---

## 3. Datasets Used

The project uses the following files:

### Historical datasets

* `wc_tournaments.csv`
  One row per World Cup edition. Contains host, champion, total goals, attendance, and format era.

* `wc_team_appearances.csv`
  One row per team per World Cup year. Contains participation status, stage reached, W/D/L/GF/GA, ELO proxy, and FIFA ranking where available.

* `wc_matches_historical.csv`
  Key historical World Cup matches such as finals, semi-finals, quarter-finals, and notable group-stage matches.

* `wc_team_alltime_stats.csv`
  Aggregated all-time World Cup statistics for each nation.

* `wc_top_scorers_by_edition.csv`
  Historical top scorer information by tournament edition.

### 2026 context datasets

* `wc_2026_groups.csv`
* `wc_2026_teams_snapshot.csv`
* `wc_prediction_features_2026.csv`
* `wc_2026_group_difficulty.csv`
* `wc_2026_qualifying_summary.csv`
* `wc_coaches_2026.csv`
* `wc_head_to_head.csv`

These files together provide team strength, coach quality, recent form, group difficulty, qualification performance, and historical matchup context.

---

## 4. Project Structure

```text
fifa_match_predictor/
├── app.py
├── requirements.txt
├── data/
│   ├── wc_tournaments.csv
│   ├── wc_team_appearances.csv
│   ├── wc_matches_historical.csv
│   ├── wc_team_alltime_stats.csv
│   ├── wc_top_scorers_by_edition.csv
│   ├── wc_2026_groups.csv
│   ├── wc_2026_teams_snapshot.csv
│   ├── wc_prediction_features_2026.csv
│   ├── wc_2026_group_difficulty.csv
│   ├── wc_2026_qualifying_summary.csv
│   ├── wc_coaches_2026.csv
│   └── wc_head_to_head.csv
├── models/
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── feature_engineering.py
    ├── train.py
    ├── predictor.py
    ├── simulator.py
    └── utils.py
```

---

## 5. High-Level Workflow

The app follows a simple end-to-end pipeline.

### Step 1: Load data

All CSVs are loaded through a central `DataLoader` module.

### Step 2: Build features

The `FeatureEngineer` creates a match-level row from team-level information.

### Step 3: Train goal models

Two regression models are trained:

* one for home goals
* one for away goals

### Step 4: Predict expected goals

For any user-selected matchup, the models estimate the expected goals for both teams.

### Step 5: Simulate the match

A Poisson-based simulation converts expected goals into scoreline probabilities.

### Step 6: Display the result

Streamlit shows the winner, scoreline, probabilities, and supporting team stats.

---

## 6. Why This Is a Regression-Based Model

A classification model would only answer:

* Team A wins
* Team B wins
* Draw

That is useful, but it does not give a scoreline.

This project instead predicts **expected goals** for each side. This is better because:

* football scores are low-count numerical outcomes,
* expected goals can be used to generate many possible results,
* scoreline probabilities are easier to compute,
* the model stays interpretable and extensible.

So the model learns a function like:

```text
features -> expected_home_goals
features -> expected_away_goals
```

Then those expected goals are converted into match probabilities.

---

## 7. Mathematics Behind the App

This section explains the logic in a mathematically grounded way.

### 7.1 Feature engineering as vector construction

For a match between Team A and Team B, the app builds a feature vector from differences and context variables.

Examples:

* ELO difference
* FIFA rank difference
* squad value difference
* qualifying goal difference
* coach experience difference
* historical head-to-head difference
* group difficulty difference

A simplified representation is:

```text
x = [elo_diff, rank_diff, form_diff, coach_exp_diff, h2h_diff, ...]
```

This vector becomes the input to the regression models.

---

### 7.2 Regression model for expected goals

The model learns two functions:

```text
ŷ_home = f_home(x)
ŷ_away = f_away(x)
```

where:

* `ŷ_home` is the predicted expected goals for the home team,
* `ŷ_away` is the predicted expected goals for the away team.

In the final implementation, tree-based regressors are used because they can learn non-linear interactions between football features.

The loss is implicitly minimized during training by the regressor’s objective, with evaluation performed using metrics such as MAE and RMSE.

---

### 7.3 Why expected goals are meaningful

Football scores are small integers, but the latent scoring ability of a team is better represented as a rate parameter.

Expected goals behave like a scoring intensity:

* higher `λ` means more likely to score more goals,
* lower `λ` means a tighter or weaker attack.

This rate then feeds the probabilistic simulation.

---

### 7.4 Poisson distribution for score simulation

A common approximation in football analytics is to model goals as Poisson-distributed random variables.

If Team A has expected goals `λ₁` and Team B has expected goals `λ₂`, then:

```text
P(Home goals = k) = e^(-λ₁) * λ₁^k / k!
P(Away goals = m) = e^(-λ₂) * λ₂^m / m!
```

The joint score probability is approximated by assuming the two scoring processes are independent:

```text
P(k, m) = P(Home goals = k) × P(Away goals = m)
```

By simulating many matches, the app estimates:

* probability Team A wins,
* probability of a draw,
* probability Team B wins,
* the most common scorelines.

This is why the app can show a likely result such as `2-1`, `1-0`, or `1-1` instead of only a winner.

---

### 7.5 Monte Carlo simulation

The simulator repeatedly samples score pairs from the Poisson distributions.

For each trial:

* sample home goals from `Poisson(λ_home)`
* sample away goals from `Poisson(λ_away)`
* compare the two scores
* count the outcome

After `N` trials:

```text
win probability = home wins / N
away probability = away wins / N
draw probability = draws / N
```

If `N = 10000`, the estimates are usually stable enough for a good front-end experience.

---

### 7.6 Confidence score

The confidence score is defined as the largest of the three match outcome probabilities:

```text
confidence = max(P_home_win, P_draw, P_away_win)
```

This gives a practical answer to the question:

> How strongly does the model prefer one outcome?

A match with `62%` win probability for one team is more confident than a match where all outcomes are close to `33%`.

---

### 7.7 Team-supporting metrics

The app also displays team statistics because predictions should not feel like a black box.

Examples include:

* current ELO
* FIFA rank
* market value
* recent form
* qualifying goal difference
* coach experience
* head-to-head record
* historical title count
* all-time World Cup win rate

These do not directly determine the answer alone; instead, they support the final prediction and make the logic more understandable.

---

## 8. Feature Logic in Detail

The feature engineering step turns multiple datasets into a single match row.

### Team-level inputs

From `wc_prediction_features_2026.csv`, the app can use:

* ELO rating
* FIFA rank
* market value
* recent form
* baseline win probability

### Qualification inputs

From `wc_2026_qualifying_summary.csv`, the app can use:

* goals for
* goals against
* goal difference
* qualification win rate
* qualification route

### Coach inputs

From `wc_coaches_2026.csv`, the app can use:

* coach age
* coach tenure
* World Cup experience
* previous best finish
* coaching style

### Head-to-head inputs

From `wc_head_to_head.csv`, the app can use:

* total meetings
* wins for Team A
* wins for Team B
* draws
* goal difference
* last meeting year
* famous historical result

### All-time strength inputs

From `wc_team_alltime_stats.csv`, the app can use:

* total appearances
* win rate
* best finish
* titles
* peak ELO approximation

The model uses these values as structured numerical evidence.

---

## 9. Why the Model Uses Differences

In football analytics, the most useful features are often not raw values but **relative values**.

For example:

* `home_elo - away_elo`
* `home_rank - away_rank`
* `home_form - away_form`
* `home_coach_exp - away_coach_exp`

This helps the model focus on the matchup itself.

The reason is simple: match outcome depends less on whether a team is strong in isolation and more on whether it is stronger than the opponent in that specific context.

---

## 10. Training Logic

The training pipeline is implemented in `src/train.py`.

### Target variables

* `home_goals`
* `away_goals`

### Training process

1. Build the historical training matrix from `wc_matches_historical.csv`.
2. Create match-level features from historical match context.
3. Split into train/test sets.
4. Train one regressor for home goals.
5. Train one regressor for away goals.
6. Evaluate the predictions on the test set.
7. Save models and feature column order to `models/`.

### Why this is done

Goal prediction is easier to generalize than direct win/loss prediction because it preserves more information about the match.

---

## 11. Prediction Logic

At inference time, when the user picks two teams in Streamlit:

1. The app extracts both teams’ profile statistics.
2. It builds a single model input row.
3. The trained models predict expected goals.
4. The simulator converts expected goals into match outcome probabilities.
5. The app picks the most likely scoreline and winner.

This gives a complete match report rather than a single label.

---

## 12. Streamlit Frontend

The Streamlit app is intentionally simple and useful.

### Main elements

* team selector for Team 1
* team selector for Team 2
* match stage selector
* host selector
* predicted winner
* predicted scoreline
* expected goals
* probabilities
* top scorelines
* supporting stats tables

### Why Streamlit

Streamlit is a good choice here because it:

* is easy to deploy,
* supports rapid iteration,
* works well with pandas and ML outputs,
* is ideal for a portfolio project.

---

## 13. How to Run the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the models

```bash
python src/train.py
```

### Start the app

```bash
streamlit run app.py
```

---

## 14. Expected Output

For a selected fixture, the app can display:

* Predicted winner
* Predicted scoreline
* Expected goals for each team
* Home win probability
* Draw probability
* Away win probability
* Confidence score
* Top scoreline probabilities
* Team comparison statistics

Example output format:

```text
Winner: Brazil
Scoreline: 2-1
Confidence: 61%
Brazil win probability: 61%
Draw probability: 22%
Germany win probability: 17%
```

---

## 15. Strengths of the Approach

* Modular and reusable code
* Uses multiple football-specific datasets
* More explainable than a black-box classifier
* Produces probabilities and scorelines
* Easy to extend with more features
* Suitable for portfolio and interview discussion

---

## 16. Limitations

Every model has limitations.

### Data limitations

* Historical match sample size is small compared to general ML datasets.
* Some features are approximate, especially pre-1993 ELO values.
* 2026-specific features are proxies and may not perfectly reflect match-day reality.

### Modeling limitations

* Poisson independence is an approximation.
* Injuries, weather, red cards, and tactical changes are not explicitly modeled.
* Scorelines can be overconfident if the regression model is very strong on a few features.

### Practical limitations

* The app is only as good as the input data.
* Tournament football has more variance than league football.

---

## 17. Possible Improvements

Future upgrades could include:

* XGBoost instead of Random Forest
* SHAP-based explanations
* Better calibration of probabilities
* Separate knockout-stage model
* Home/neutral venue adjustment
* Injury and squad-availability features
* Bayesian score model
* Tournament bracket simulator
* GraphRAG for historical football Q&A

---

## 18. Why This Is a Good Portfolio Project

This project demonstrates:

* end-to-end ML workflow
* feature engineering
* regression modeling
* probabilistic simulation
* explainability
* modular software design
* web app development
* sports analytics thinking

It is a strong project for a data science or machine learning portfolio because it combines theory with a usable product.

---

## 19. File Responsibilities

### `src/data_loader.py`

Loads all CSV files from the data folder in a single, centralized place.

### `src/feature_engineering.py`

Constructs model-ready features for historical training and user predictions.

### `src/train.py`

Trains the regression models and saves the trained artifacts.

### `src/predictor.py`

Loads the saved models and produces predictions for a selected match.

### `src/simulator.py`

Uses Poisson simulation to estimate probabilities and scorelines.

### `src/utils.py`

Contains reusable helper functions for normalization, scoring, and formatting.

### `app.py`

Provides the Streamlit UI and connects everything together.

---

## 20. Final Summary

This app is a **World Cup match prediction engine** built from multiple football datasets and powered by a regression-plus-simulation approach.

It predicts:

* who is likely to win,
* what the scoreline could be,
* how confident the model is,
* and why the prediction was made.

The design keeps the logic modular, interpretable, and easy to extend into a more advanced football analytics system.
