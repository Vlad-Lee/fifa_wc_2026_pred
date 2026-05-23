# FIFA World Cup 2026 — Prediction Model: Project Overview

> **Tournament dates:** June 11 – July 19, 2026 | 48 teams | 104 matches

---

## 1. What Are We Predicting?

This project focuses on two target variables:

| Target | Output | Use Case |
|--------|--------|----------|
| **Goals per team** | Goals scored (Poisson count) | Scoreline distribution, xG comparison |
| **Match outcome** | Win / Draw / Loss (derived) | Bracket simulation, tournament odds |

Goals per team is the *primary* target — we model each team's goals independently using Dixon-Coles Poisson regression, then derive match outcome probabilities directly from the scoreline distribution. This means outcome prediction comes for free once the goal model is working: sum up all (i, j) scoreline probabilities where i > j for a home win, i == j for a draw, and i < j for an away win.

---

## 2. The Dixon-Coles Model

### 2.1 Core Framework

The project is built entirely around the **Dixon-Coles Poisson model**. The pipeline flows like this:

1. **Team strength estimation** — fit attack and defense ratings for every team via MLE on historical results
2. **Expected goals (λ) prediction** — given two teams, compute λ_A and λ_B (expected goals for each)
3. **Scoreline distribution** — apply the Dixon-Coles correction to get P(Goals_A = i, Goals_B = j) for all i, j
4. **Outcome derivation** — sum scoreline probabilities to get P(Win), P(Draw), P(Loss) with no extra model needed
5. **Tournament simulation** — sample scorelines in a Monte Carlo loop to simulate the full bracket

---

### 2.2 How Dixon-Coles Works

Goals for each team are modeled as independent Poisson random variables:

```
Goals_A ~ Poisson(λ_A)     where λ_A = exp(intercept + att_A + def_B)
Goals_B ~ Poisson(λ_B)     where λ_B = exp(intercept + att_B + def_A)
```

The **τ (tau) correction factor** adjusts the joint probability for low-scoring outcomes (0-0, 1-0, 0-1, 1-1), which occur more or less often than pure Poisson predicts:

```python
def tau(goals_a, goals_b, lambda_a, lambda_b, rho):
    if goals_a == 0 and goals_b == 0:
        return 1 - lambda_a * lambda_b * rho
    elif goals_a == 0 and goals_b == 1:
        return 1 + lambda_a * rho
    elif goals_a == 1 and goals_b == 0:
        return 1 + lambda_b * rho
    elif goals_a == 1 and goals_b == 1:
        return 1 - rho
    else:
        return 1.0

# Full scoreline probability:
P(i, j) = poisson.pmf(i, lambda_a) * poisson.pmf(j, lambda_b) * tau(i, j, lambda_a, lambda_b, rho)
```

**Parameters estimated via MLE:**
- `att_team` — attack strength for each team (relative to a sum-to-zero constraint)
- `def_team` — defense strength for each team (negative value = strong defense)
- `intercept` — baseline scoring rate across all matches
- `rho` — low-score correction (typically small and negative, ~−0.1)

**Time-decay weighting** — recent matches should count more:
```python
weight = exp(-xi * days_since_match)   # xi ≈ 0.0018 gives a half-life of ~385 days
```
This is passed as sample weights into the log-likelihood during MLE.

---

### 2.3 Deriving Outcome Probabilities

Once you have the scoreline matrix, outcomes drop out naturally:

```python
score_matrix = np.zeros((11, 11))
for i in range(11):
    for j in range(11):
        score_matrix[i, j] = poisson.pmf(i, lam_a) * poisson.pmf(j, lam_b) * tau(i, j, ...)

p_win  = np.triu(score_matrix, k=1).sum()   # goals_A > goals_B
p_draw = np.diag(score_matrix).sum()        # goals_A == goals_B
p_loss = np.tril(score_matrix, k=-1).sum()  # goals_A < goals_B

# Expected goals for each team (for reporting):
exp_goals_a = sum(i * score_matrix[i, :].sum() for i in range(11))
exp_goals_b = sum(j * score_matrix[:, j].sum() for j in range(11))
```

---

## 3. Free Data Sources

### Historical Match Results (the backbone)
| Source | URL | What You Get |
|--------|-----|-------------|
| **football-data.co.uk** | football-data.co.uk | Match results, odds, going back decades |
| **OpenFootball (GitHub)** | github.com/openfootball | All WC results historically, JSON format |
| **Kaggle — International Football Results** | kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017 | 45k+ international matches since 1872 |
| **SportsReference / FBref** | fbref.com | Advanced stats, xG, player-level data |
| **WorldFootball.net** | worldfootball.net | Comprehensive historical results |

### Ratings & Rankings
| Source | URL | What You Get |
|--------|-----|-------------|
| **FIFA World Rankings** | fifa.com/fifa-world-ranking | Official FIFA points (updated monthly) |
| **ClubElo** | clubelo.com | Club-level Elo (useful for player proxies) |
| **EloRatings.net** | eloratings.net | National team Elo ratings, downloadable CSV |
| **FiveThirtyEight SPI** | github.com/fivethirtyeight/data | Club SPI data (archived; WC projections) |

### APIs (free tier available)
| Source | URL | Notes |
|--------|-----|-------|
| **API-Football** | api-football.com | 100 free requests/day; fixtures, odds, stats |
| **Football-Data.org** | football-data.org | Free tier; competitions, standings, lineups |
| **OpenLigaDB** | openligadb.de | German league focused, but free & open |

### Player / Squad Data
| Source | URL | What You Get |
|--------|-----|-------------|
| **Transfermarkt** | transfermarkt.com | Squad values, player ages, transfer data |
| **StatsBomb Open Data** | github.com/statsbomb/open-data | Event-level data (360° tracking) for select competitions |
| **SofaScore** | sofascore.com | Player ratings, lineup data |

### Betting Odds (as market probabilities)
| Source | URL | Notes |
|--------|-----|-------|
| **The Odds API** | the-odds-api.com | 500 free requests/month; consensus odds |
| **football-data.co.uk** | football-data.co.uk | Historical odds from major bookmakers |

---

## 4. End-to-End Project Workflow

```
Phase 1: Define           Phase 2: Data            Phase 3: Features
─────────────────         ──────────────────        ──────────────────
Choose target variable  → Collect & store data    → Engineer features
Define evaluation metric  Clean & validate          Build team strength ratings
Plan tournament sim       EDA                       Rolling window stats
```
```
Phase 4: Model            Phase 5: Validate        Phase 6: Simulate
─────────────────         ──────────────────        ──────────────────
Train base models       → Time-series CV           → Monte Carlo bracket sim
Tune hyperparameters      Calibration check          10k+ tournament runs
Build ensemble            Backtesting on past WCs    Group stage + knockouts
```

---

### Phase 1 — Problem Definition

**Decision 1: Target variable**
- Recommended: model goals for each team (Poisson regression), then derive everything else

**Decision 2: Match scope**
- Only FIFA World Cup matches? Or all international A matches? (More data → better estimates but different contexts)
- Recommended: train on all senior international matches, weight recent matches more heavily

**Decision 3: Evaluation metric**
- For outcome classification: log-loss, Brier score (probability calibration matters more than accuracy)
- For goal prediction: MAE, RMSE on goal totals
- For tournament simulation: compare bracket completion probabilities to betting market implied odds

---

### Phase 2 — Data Collection & Storage

```
data/
├── raw/
│   ├── international_results.csv     # From Kaggle / football-data.co.uk
│   ├── fifa_rankings.csv             # Monthly snapshots
│   ├── elo_ratings.csv               # From eloratings.net
│   └── wc_2026_fixtures.csv          # Tournament bracket
├── processed/
│   ├── matches_features.parquet
│   └── team_strength_ratings.parquet
└── external/
    └── squad_values.csv              # From Transfermarkt
```

Key steps:
- Standardize team names (this is the biggest headache — "USA", "United States", "US" are all the same team)
- Build a canonical team name lookup table early
- Store timestamps so you can reconstruct the "state of the world" at any past date (crucial for avoiding data leakage)

---

### Phase 3 — Feature Engineering

Dixon-Coles is parameter-lean by design — the core model only needs match results (goals for/against, date, teams). The main feature work is controlling *which matches* feed the MLE and how they're weighted.

**Time decay** — the most important lever:
```python
import numpy as np

def time_weight(match_date, reference_date, xi=0.0018):
    """Exponential decay — xi=0.0018 gives ~385-day half-life."""
    days_elapsed = (reference_date - match_date).days
    return np.exp(-xi * days_elapsed)
```

**Match filtering** — decide which historical matches to include:
- All senior international A matches (recommended for sample size)
- Optional: down-weight friendlies vs. competitive matches (add a `match_weight` multiplier)
- Cut-off: matches older than ~8 years contribute negligible weight anyway

**Constraint** — attack and defense parameters need an identifiability constraint to avoid unbounded solutions:
```python
# Sum-to-zero: average attack and defense across all teams = 0
# Enforce via fixing one team's parameter or adding a penalty term
constraints = {"type": "eq", "fun": lambda x: sum(x[attack_indices])}
```

**Avoid data leakage:**
- Always use only matches played *before* the match being predicted
- When backtesting, refit the model at each evaluation date — don't fit on the full history

---

### Phase 4 — Model Training

The Dixon-Coles model is fit by **maximum likelihood estimation (MLE)** using `scipy.optimize.minimize`.

```python
from scipy.optimize import minimize
from scipy.stats import poisson

def neg_log_likelihood(params, matches, teams, xi=0.0018):
    """Negative log-likelihood of all matches given team parameters."""
    att, defe, intercept, rho = unpack_params(params, teams)
    total_ll = 0.0
    for match in matches:
        lam_a = np.exp(intercept + att[match.home] + defe[match.away])
        lam_b = np.exp(intercept + att[match.away] + defe[match.home])
        t = tau(match.goals_a, match.goals_b, lam_a, lam_b, rho)
        ll = (np.log(poisson.pmf(match.goals_a, lam_a))
            + np.log(poisson.pmf(match.goals_b, lam_b))
            + np.log(t))
        weight = time_weight(match.date, reference_date, xi)
        total_ll += weight * ll
    return -total_ll

result = minimize(
    neg_log_likelihood,
    x0=initial_params,
    args=(matches, teams),
    method="L-BFGS-B",
    constraints=sum_to_zero_constraint,
)
```

**Hyperparameters to tune:**
- `xi` — time decay rate (tune on validation log-loss; try 0.001–0.003)
- `rho` — estimated jointly with attack/defense params via MLE
- Match inclusion cutoff — how many years of history to include

---

### Phase 5 — Validation

**Critical: time-series cross-validation (no shuffling!)**
```
Train: 2006–2014 WC matches → Validate: 2018 WC
Train: 2006–2018 WC matches → Validate: 2022 WC
```

For non-WC training data, use expanding window CV with a gap to prevent leakage.

**Calibration check:**
- Plot predicted probability vs. actual frequency (reliability diagram)
- Well-calibrated model: if you say a team has 30% win probability, they should win ~30% of those matches
- Apply Platt scaling or isotonic regression if poorly calibrated

**Backtesting:**
- Simulate the 2018 and 2022 World Cups using your model
- Compare your bracket probabilities to the betting market (Pinnacle, Betfair closing odds are the gold standard)
- Target: log-loss ≤ market log-loss on held-out WC matches

---

### Phase 6 — Tournament Simulation

```python
import numpy as np
from itertools import product

def simulate_match(team_a, team_b, model):
    """Returns (goals_a, goals_b) sampled from model distribution."""
    lambda_a, lambda_b = model.predict(team_a, team_b)
    return np.random.poisson(lambda_a), np.random.poisson(lambda_b)

def simulate_tournament(bracket, model, n_sims=10_000):
    results = defaultdict(int)
    for _ in range(n_sims):
        winner = run_single_tournament(bracket, model, simulate_match)
        results[winner] += 1
    return {team: count / n_sims for team, count in results.items()}
```

**Handle knockouts carefully:**
- In knockout matches with no winner after 90 min, add extra time + penalties
- Penalty shootout can be modeled as a coin flip or with historical penalty data (~53% for team that scores first)
- Track: group stage finish positions, round of 32/16/QF/SF/F/winner probabilities

**Output:**
```
Team              | Win %  | Final % | SF %   | QF %
──────────────────|────────|─────────|────────|──────
Brazil            | 12.4%  | 21.3%   | 38.2%  | 58.1%
France            | 10.8%  | 19.7%   | 36.0%  | 55.4%
England           |  9.1%  | 17.2%   | 32.8%  | 51.9%
...
```

---

## 5. Recommended Tech Stack

| Layer | Tool |
|-------|------|
| Data wrangling | `pandas` |
| Numerical / optimization | `numpy`, `scipy.optimize` |
| Statistics | `scipy.stats` (Poisson PMF/CDF) |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Notebooks | `jupyter` |
| Data storage | Parquet files (`pyarrow`) or CSV |

---

## 6. Suggested Project Milestones

| Week | Milestone |
|------|-----------|
| **Week 1** | Data collection, cleaning, team name normalization |
| **Week 2** | EDA, time-decay weighting, initial MLE parameter fit |
| **Week 3** | Full Dixon-Coles model, scoreline matrix, outcome probabilities |
| **Week 4** | Backtesting on 2018 + 2022 WCs, calibration check |
| **Week 5** | Monte Carlo tournament simulation, final bracket output |
| **WC Kickoff (Jun 11)** | Live predictions, re-fit model after each matchday |

---

## 7. Key Pitfalls to Avoid

1. **Data leakage** — using future information in past predictions invalidates your backtest
2. **Team name inconsistency** — "Ivory Coast", "Côte d'Ivoire", "Cote d'Ivoire" are the same team; build a lookup table early
3. **Over-fitting on small WC samples** — the World Cup only happens every 4 years; supplement with all international matches
4. **Ignoring calibration** — a model that predicts "100% win" for favorites will look good on accuracy but is useless for simulation
5. **Treating penalties as predictable** — knockout outcomes are genuinely noisy; don't overfit to past results
6. **Not updating the model** — as the tournament progresses, refit or update ratings with new match data

---

*Last updated: May 2026 | WC 2026 kicks off June 11 in Mexico City*
