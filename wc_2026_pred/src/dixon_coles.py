"""
dixon_coles.py
==============

A full implementation of the Dixon-Coles (1997) football prediction model,
with exponential time-decay weighting, fitted to the international match
results produced by the existing cleaning pipeline (model_data_cleaning.py).

WHAT THE MODEL DOES, IN ONE PARAGRAPH
-------------------------------------
For a match, each team scores a number of goals drawn from a Poisson
distribution. The mean of that distribution depends on the *attacking*
strength of the team and the *defensive* strength of its opponent, plus a
*home-advantage* bonus for the home side. Dixon and Coles added two refinements
on top of this: (1) a low-score correction `tau` that fixes the well-known
tendency of the independent-Poisson model to misprice 0-0, 1-0, 0-1 and 1-1
results, and (2) time-decay weighting so that recent matches influence the fit
more than old ones. All parameters are estimated by maximum likelihood.

PIPELINE POSITION
-----------------
    model_data_cleaning.py  ->  modeling_matrices_package.zip  (train/test)
    elo.py                  ->  (adds Elo columns; not needed here)
    dixon_coles.py          ->  dixon_coles_params.json        <-- THIS FILE
    simulate_wc2026.py      ->  uses dixon_coles_params.json

Run it directly:  python src/dixon_coles.py
"""

import json
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Date the predictions are "as of". Time-decay weights are measured backwards
# from here, so a match played the day before this date is barely discounted,
# while one played years earlier is discounted heavily.
PREDICTION_DATE = "2026-06-11"          # 2026 World Cup opening match

# Exponential decay rate, per day. weight = exp(-XI * age_in_days).
# XI = 0.0018 gives a half-life of ln(2)/0.0018 ~= 385 days, i.e. a match
# loses half its influence roughly every year. XI = 0 disables decay entirely.
# Use tune_decay() to choose this empirically instead of guessing.
XI = 0.0018

# Ignore matches before this date. Older games carry near-zero decay weight
# anyway, so dropping them just makes the fit faster without changing results.
MIN_DATE = "2014-01-01"

# L2 (ridge) regularisation strength on the attack/defence parameters. This
# gently shrinks teams with very little recent data toward the global average,
# which stabilises minnows (e.g. Curaçao, Haiti) that play few matches.
RIDGE = 0.01

# Upper bound on goals when building a score-probability matrix. Scorelines
# above this are so rare they are safely ignored.
MAX_GOALS = 10


# ===========================================================================
# STEP 1 — LOAD THE MATCH DATA
# ===========================================================================
def load_matches(zip_path, use_all=True):
    """
    Load completed matches from the modelling package produced upstream.

    Parameters
    ----------
    zip_path : Path
        Path to modeling_matrices_package.zip (train.csv / test.csv inside).
    use_all : bool
        True  -> concatenate train + test (use this for the real WC forecast,
                 you want every match up to today).
        False -> train split only (use this when you are evaluating the model
                 against the held-out test period).

    Returns
    -------
    DataFrame with columns: date, home_team, away_team, home_score,
    away_score, neutral.
    """
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        names = ["train.csv"] + (["test.csv"] if use_all else [])
        for name in names:
            with z.open(name) as f:
                frames.append(pd.read_csv(f, parse_dates=["date"]))

    df = pd.concat(frames, ignore_index=True)

    # Keep only the columns the model needs and coerce types defensively
    # (CSV round-trips can turn the boolean 'neutral' column into strings).
    df = df[["date", "home_team", "away_team",
             "home_score", "away_score", "neutral"]].copy()
    df["neutral"] = (
        df["neutral"].astype(str).str.strip().str.lower().isin(["true", "1"])
    )
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df = df[df["date"] >= pd.Timestamp(MIN_DATE)].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ===========================================================================
# STEP 2 — TURN THE DATA INTO NUMERIC ARRAYS
# ===========================================================================
def build_design(df, prediction_date, xi):
    """
    Convert the match DataFrame into the integer/array form the optimiser
    needs, and compute one time-decay weight per match.

    Returns a dict with:
        teams      : list of team names, position = parameter index
        home_idx   : int array, index of the home team for each match
        away_idx   : int array, index of the away team
        home_goals : int array
        away_goals : int array
        home_flag  : float array, 1.0 if a real home game, 0.0 if neutral
        weights    : float array, exp(-xi * age_in_days)
    """
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    team_index = {t: i for i, t in enumerate(teams)}

    home_idx = df["home_team"].map(team_index).to_numpy()
    away_idx = df["away_team"].map(team_index).to_numpy()
    home_goals = df["home_score"].to_numpy(dtype=float)
    away_goals = df["away_score"].to_numpy(dtype=float)

    # Home advantage only applies when the match is actually played at the
    # home team's venue. At a neutral venue the "home team" is just whichever
    # side the dataset happened to list first, so it gets no bonus.
    home_flag = (~df["neutral"].to_numpy()).astype(float)

    # Time-decay weight. age is how many days before PREDICTION_DATE the match
    # was played; clip at 0 so a (hypothetical) future match is never upweighted.
    pred = pd.Timestamp(prediction_date)
    age_days = (pred - df["date"]).dt.days.to_numpy().astype(float)
    age_days = np.clip(age_days, 0.0, None)
    weights = np.exp(-xi * age_days)

    return {
        "teams": teams,
        "home_idx": home_idx,
        "away_idx": away_idx,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_flag": home_flag,
        "weights": weights,
    }


# ===========================================================================
# STEP 3 — THE DIXON-COLES LOW-SCORE CORRECTION (tau)
# ===========================================================================
def dixon_coles_tau(home_goals, away_goals, lam, mu, rho):
    """
    The tau adjustment. A plain "two independent Poissons" model gets the
    frequency of low scores slightly wrong — in particular it understates
    draws. Dixon & Coles multiply the joint probability of the four lowest
    scorelines by a correction factor controlled by a single parameter rho:

        (0,0) -> 1 - lam*mu*rho
        (0,1) -> 1 + lam*rho
        (1,0) -> 1 + mu*rho
        (1,1) -> 1 - rho
        all other scorelines -> 1  (unchanged)

    A negative rho (the empirically usual case) pushes probability toward
    0-0 and 1-1 and away from 1-0 and 0-1, i.e. it makes draws a bit more
    likely, matching real data.

    This function is vectorised: every argument is a NumPy array.
    """
    tau = np.ones_like(lam, dtype=float)

    is_00 = (home_goals == 0) & (away_goals == 0)
    is_01 = (home_goals == 0) & (away_goals == 1)
    is_10 = (home_goals == 1) & (away_goals == 0)
    is_11 = (home_goals == 1) & (away_goals == 1)

    tau = np.where(is_00, 1.0 - lam * mu * rho, tau)
    tau = np.where(is_01, 1.0 + lam * rho, tau)
    tau = np.where(is_10, 1.0 + mu * rho, tau)
    tau = np.where(is_11, 1.0 - rho, tau)
    return tau


# ===========================================================================
# STEP 4 — THE (NEGATIVE) LOG-LIKELIHOOD
# ===========================================================================
def negative_log_likelihood(params, design, ridge):
    """
    The objective the optimiser minimises.

    The parameter vector `params` is laid out as:
        [ attack_0 ... attack_{n-1},
          defence_0 ... defence_{n-1},
          home_advantage,
          rho ]

    For a match between home team i and away team j:
        log(lambda) = attack_i + defence_j + home_advantage * home_flag
        log(mu)     = attack_j + defence_i
    where lambda is the home team's expected goals and mu the away team's.

    The contribution of one match to the log-likelihood is:
        log(tau) + [home_goals*log(lambda) - lambda]
                 + [away_goals*log(mu)     - mu]
    (the Poisson factorial term is a constant and is dropped — it does not
    affect where the maximum is).

    Each match is multiplied by its time-decay weight, so recent matches
    pull on the parameters harder than old ones.
    """
    n = len(design["teams"])

    attack = params[:n]
    # IDENTIFIABILITY: the model is unchanged if you add a constant to every
    # attack value and subtract it from every defence value. We remove that
    # ambiguity by forcing the attack parameters to average to zero.
    attack = attack - attack.mean()
    defence = params[n:2 * n]
    home_adv = params[2 * n]
    rho = params[2 * n + 1]

    hi = design["home_idx"]
    ai = design["away_idx"]
    hg = design["home_goals"]
    ag = design["away_goals"]
    w = design["weights"]

    log_lam = attack[hi] + defence[ai] + home_adv * design["home_flag"]
    log_mu = attack[ai] + defence[hi]
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)

    tau = dixon_coles_tau(hg, ag, lam, mu, rho)
    # For extreme rho the correction can go non-positive; floor it so log()
    # stays finite. The optimiser then naturally avoids that region.
    tau = np.clip(tau, 1e-12, None)

    log_lik = (
        np.log(tau)
        + hg * log_lam - lam
        + ag * log_mu - mu
    )

    weighted = np.sum(w * log_lik)

    # Ridge penalty: a gentle pull of attack/defence toward 0 (the average
    # team). Stabilises teams with little recent data. Not part of classic
    # Dixon-Coles — set RIDGE = 0 to switch it off.
    penalty = ridge * (np.sum(attack ** 2) + np.sum(defence ** 2))

    return -weighted + penalty


# ===========================================================================
# STEP 5 — FIT THE MODEL (MAXIMUM LIKELIHOOD)
# ===========================================================================
def fit(design, ridge=RIDGE, verbose=True):
    """
    Estimate every parameter by minimising the negative log-likelihood with
    L-BFGS-B. Returns a fitted DixonColesModel.
    """
    n = len(design["teams"])

    # Starting point: all teams average (0), a mild home advantage, a small
    # negative rho (its usual sign).
    x0 = np.concatenate([
        np.zeros(n),          # attack
        np.zeros(n),          # defence
        [0.25],               # home advantage
        [-0.05],              # rho
    ])

    # Bounds keep the search in a sensible, numerically safe region.
    bounds = (
        [(-3.0, 3.0)] * n +   # attack
        [(-3.0, 3.0)] * n +   # defence
        [(-0.5, 1.5)] +       # home advantage
        [(-0.2, 0.2)]         # rho
    )

    if verbose:
        print(f"  Fitting {2 * n + 2} parameters on "
              f"{len(design['home_idx']):,} matches ...")

    result = minimize(
        negative_log_likelihood,
        x0,
        args=(design, ridge),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "maxfun": 200000},
    )

    if verbose:
        status = "converged" if result.success else f"STOPPED ({result.message})"
        print(f"  Optimiser {status}; final -log L = {result.fun:,.1f}")

    p = result.x
    attack = p[:n] - p[:n].mean()      # apply the same centring as the objective
    defence = p[n:2 * n]

    return DixonColesModel(
        teams=design["teams"],
        attack=attack,
        defence=defence,
        home_advantage=float(p[2 * n]),
        rho=float(p[2 * n + 1]),
    )


# ===========================================================================
# STEP 6 — THE FITTED MODEL OBJECT
# ===========================================================================
class DixonColesModel:
    """
    Holds the fitted parameters and turns them into match predictions.

    The simulation script imports this class, loads a saved model with
    DixonColesModel.load(...), and calls score_matrix() / sample_score().
    """

    def __init__(self, teams, attack, defence, home_advantage, rho,
                 xi=XI, prediction_date=PREDICTION_DATE):
        self.attack = dict(zip(teams, np.asarray(attack, dtype=float)))
        self.defence = dict(zip(teams, np.asarray(defence, dtype=float)))
        self.home_advantage = float(home_advantage)
        self.rho = float(rho)
        self.xi = float(xi)
        self.prediction_date = prediction_date

    # ---- expected goals -------------------------------------------------
    def expected_goals(self, home, away, neutral=True, home_factor=None):
        """
        Return (lambda, mu): the expected goals for the home and away team.

        `home_factor` (0.0-1.0) is the fraction of the fitted home-advantage
        term to apply to the home side — 0.0 for a neutral venue, 1.0 for a
        full home game, 0.5 for a partial edge such as a World Cup host. If it
        is left as None it falls back to the `neutral` flag: neutral=True
        gives 0.0, neutral=False gives 1.0.
        """
        for t in (home, away):
            if t not in self.attack:
                raise KeyError(
                    f"Team '{t}' is not in the fitted model. Check the "
                    f"spelling against results.csv (see NAME_ALIASES)."
                )
        if home_factor is None:
            home_factor = 0.0 if neutral else 1.0
        home_term = home_factor * self.home_advantage
        lam = np.exp(self.attack[home] + self.defence[away] + home_term)
        mu = np.exp(self.attack[away] + self.defence[home])
        return lam, mu

    # ---- full scoreline distribution ------------------------------------
    def score_matrix(self, home, away, neutral=True,
                     max_goals=MAX_GOALS, rate_scale=1.0, home_factor=None):
        """
        Return an (max_goals+1) x (max_goals+1) matrix M where
        M[x, y] = P(home scores x, away scores y).

        `rate_scale` multiplies both expected-goal rates. It is used by the
        simulation to model a 30-minute period of extra time, where goals
        accrue for only one third of a 90-minute match (rate_scale = 1/3).

        `home_factor` is passed straight through to expected_goals() — see
        there for how partial home advantage works.
        """
        lam, mu = self.expected_goals(home, away, neutral, home_factor)
        lam *= rate_scale
        mu *= rate_scale

        goals = np.arange(0, max_goals + 1)
        # Independent-Poisson grid: outer product of the two marginal pmfs.
        home_pmf = poisson.pmf(goals, lam)
        away_pmf = poisson.pmf(goals, mu)
        matrix = np.outer(home_pmf, away_pmf)

        # Apply the Dixon-Coles tau correction to the four low-score cells.
        matrix[0, 0] *= 1.0 - lam * mu * self.rho
        matrix[0, 1] *= 1.0 + lam * self.rho
        matrix[1, 0] *= 1.0 + mu * self.rho
        matrix[1, 1] *= 1.0 - self.rho

        # tau slightly breaks normalisation; clip negatives and renormalise.
        matrix = np.clip(matrix, 0.0, None)
        matrix /= matrix.sum()
        return matrix

    def outcome_probabilities(self, home, away, neutral=True):
        """Return dict with P(home win), P(draw), P(away win)."""
        m = self.score_matrix(home, away, neutral)
        return {
            "home_win": float(np.tril(m, -1).sum()),  # home goals > away goals
            "draw": float(np.trace(m)),
            "away_win": float(np.triu(m, 1).sum()),   # away goals > home goals
        }

    # ---- persistence ----------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": "dixon-coles",
            "fitted_at": datetime.now().isoformat(timespec="seconds"),
            "prediction_date": self.prediction_date,
            "xi": self.xi,
            "home_advantage": self.home_advantage,
            "rho": self.rho,
            # Cast NumPy floats to plain Python floats so json can serialise them.
            "attack": {t: float(v) for t, v in self.attack.items()},
            "defence": {t: float(v) for t, v in self.defence.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        teams = list(d["attack"].keys())
        model = cls(
            teams=teams,
            attack=[d["attack"][t] for t in teams],
            defence=[d["defence"][t] for t in teams],
            home_advantage=d["home_advantage"],
            rho=d["rho"],
            xi=d.get("xi", XI),
            prediction_date=d.get("prediction_date", PREDICTION_DATE),
        )
        return model


# ===========================================================================
# STEP 7 (OPTIONAL) — TUNE THE TIME-DECAY RATE
# ===========================================================================
def tune_decay(zip_path, xi_grid=(0.0, 0.0008, 0.0014, 0.0018, 0.0026, 0.004)):
    """
    Choose XI empirically. For each candidate decay rate we fit the model on
    the training split and then measure the average log-likelihood it assigns
    to the held-out test matches. The XI with the best test-set score wins.

    This does one full fit per grid point, so it is the slow path — run it
    once, note the best XI, then hard-code it in the CONFIGURATION block.
    """
    # Train on the pre-2022 split; evaluate forecasts on the 2022+ test split.
    train_df = load_matches(zip_path, use_all=False)
    with zipfile.ZipFile(zip_path) as z:
        with z.open("test.csv") as f:
            test_df = pd.read_csv(f, parse_dates=["date"])
    test_df["neutral"] = (
        test_df["neutral"].astype(str).str.lower().isin(["true", "1"])
    )

    print("Tuning time-decay rate XI:")
    scores = {}
    for xi in xi_grid:
        design = build_design(train_df, PREDICTION_DATE, xi)
        model = fit(design, verbose=False)
        # Average log-likelihood per test match (higher = better forecasts).
        total, count = 0.0, 0
        for _, row in test_df.iterrows():
            try:
                m = model.score_matrix(row["home_team"], row["away_team"],
                                       neutral=bool(row["neutral"]))
            except KeyError:
                continue  # team unseen in training — skip
            x, y = int(row["home_score"]), int(row["away_score"])
            if x <= MAX_GOALS and y <= MAX_GOALS:
                total += np.log(max(m[x, y], 1e-12))
                count += 1
        scores[xi] = total / max(count, 1)
        print(f"  XI = {xi:<8} mean test log-lik = {scores[xi]:.4f}")

    best = max(scores, key=scores.get)
    print(f"Best XI = {best}")
    return best, scores


# ===========================================================================
# STEP 8 — DIAGNOSTICS
# ===========================================================================
def print_diagnostics(model, top_n=20):
    """Print a human-readable summary so you can sanity-check the fit."""
    print("\n" + "=" * 60)
    print("FITTED DIXON-COLES PARAMETERS")
    print("=" * 60)
    print(f"Home advantage : {model.home_advantage:+.3f}  "
          f"(home team's expected goals multiplied by "
          f"{np.exp(model.home_advantage):.2f})")
    print(f"rho (low-score): {model.rho:+.3f}  "
          f"({'draws upweighted' if model.rho < 0 else 'draws downweighted'})")

    # Overall team rating = attack - defence. Higher attack is better; a more
    # NEGATIVE defence value is better (it suppresses the opponent), so the
    # natural single-number strength index is attack minus defence.
    strength = {t: model.attack[t] - model.defence[t] for t in model.attack}
    ranked = sorted(strength.items(), key=lambda kv: kv[1], reverse=True)

    print(f"\nTop {top_n} teams by overall strength (attack - defence):")
    print(f"  {'Team':<26}{'Attack':>9}{'Defence':>9}{'Strength':>10}")
    for team, s in ranked[:top_n]:
        print(f"  {team:<26}{model.attack[team]:>+9.3f}"
              f"{model.defence[team]:>+9.3f}{s:>+10.3f}")


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    zip_path = base_dir / "data" / "processed" / "modeling" / \
        "modeling_matrices_package.zip"
    out_path = base_dir / "data" / "processed" / "modeling" / \
        "dixon_coles_params.json"

    print("Loading match data ...")
    matches = load_matches(zip_path, use_all=True)
    print(f"  {len(matches):,} matches from {matches['date'].min().date()} "
          f"to {matches['date'].max().date()}")

    print("Building design matrix and time-decay weights ...")
    design = build_design(matches, PREDICTION_DATE, XI)
    print(f"  {len(design['teams'])} teams")

    print("Fitting Dixon-Coles model ...")
    model = fit(design)

    print_diagnostics(model)

    model.save(out_path)
    print(f"\nSaved fitted parameters to {out_path}")

    # Quick demo prediction (neutral venue, as at a World Cup).
    for home, away in [("Spain", "France"), ("Brazil", "Scotland")]:
        try:
            probs = model.outcome_probabilities(home, away, neutral=True)
            lam, mu = model.expected_goals(home, away, neutral=True)
            print(f"\n{home} vs {away} (neutral):")
            print(f"  expected goals: {lam:.2f} - {mu:.2f}")
            print(f"  P({home} win) = {probs['home_win']:.1%} | "
                  f"draw = {probs['draw']:.1%} | "
                  f"P({away} win) = {probs['away_win']:.1%}")
        except KeyError as e:
            print(f"  (skipped demo: {e})")
