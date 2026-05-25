"""
lib.py - shared helpers for the 2026 World Cup Streamlit app.

Centralises everything the pages need: locating the project, loading the
fitted Dixon-Coles model, running and caching the tournament simulation,
building a projected ("most likely") bracket, and the common page chrome.
Keeping it here lets every page stay short.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Locate the project and make the src/ modules importable.
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_DIR = BASE_DIR / "data" / "processed" / "modeling"
PARAMS_PATH = DATA_DIR / "dixon_coles_params.json"
PRED_PATH = DATA_DIR / "wc2026_predictions.csv"

import dixon_coles as dc          # noqa: E402  (import after sys.path tweak)
import wc2026_config as cfg       # noqa: E402
import simulate_wc2026 as sim     # noqa: E402

# Probability columns the simulation produces.
STAGE_COLS = ["P(advance)", "P(round of 16)", "P(quarter-final)",
              "P(semi-final)", "P(final)", "P(champion)"]
FINISH_COLS = ["P(1st)", "P(2nd)", "P(3rd)", "P(4th)"]


# ===========================================================================
# MODEL
# ===========================================================================
@st.cache_resource(show_spinner=False)
def get_model():
    """Load the fitted Dixon-Coles model once per session. None if not fitted."""
    if not PARAMS_PATH.exists():
        return None
    return dc.DixonColesModel.load(PARAMS_PATH)


@st.cache_data(show_spinner=False)
def model_info():
    """Lightweight metadata for the sidebar. None if the model is missing."""
    if not PARAMS_PATH.exists():
        return None
    with open(PARAMS_PATH, encoding="utf-8") as f:
        d = json.load(f)
    return {
        "fitted_at": d.get("fitted_at", "unknown"),
        "prediction_date": d.get("prediction_date", "unknown"),
        "xi": d.get("xi"),
        "home_advantage": d.get("home_advantage"),
        "rho": d.get("rho"),
        "n_teams": len(d.get("attack", {})),
    }


def ensure_model():
    """Return the model, or stop the page with a friendly message."""
    model = get_model()
    if model is None:
        st.error(
            "No fitted model found. Run `python src/dixon_coles.py` to create "
            "`dixon_coles_params.json`, then reload this app."
        )
        st.stop()
    return model


# ===========================================================================
# SIMULATION
# ===========================================================================
@st.cache_data(show_spinner=False, persist="disk")
def run_simulation(n_sims, host_fraction):
    """
    Run the tournament simulation. Cached by (n_sims, host_fraction).

    persist="disk" means Streamlit also writes the cached result to its own
    on-disk cache, so re-running the same settings is instant even after the
    app restarts - no hand-managed CSV needed. The cache is process-global, so
    one run is shared by every user of the app.
    """
    model = get_model()
    sim.HOST_ADVANTAGE_FRACTION = float(host_fraction)
    sim.NAME_MAP = sim.resolve_team_names(model)
    sim._cdf_cache.clear()
    return sim.run(model, int(n_sims))


@st.cache_data(show_spinner=False)
def load_precomputed():
    """Load a previously saved predictions CSV, or None."""
    if PRED_PATH.exists():
        return pd.read_csv(PRED_PATH)
    return None


def get_predictions():
    """Current predictions: a live run from this session, else the saved CSV."""
    if "predictions" not in st.session_state:
        pre = load_precomputed()
        if pre is not None:
            st.session_state["predictions"] = pre
            st.session_state["sim_meta"] = {
                "source": "saved file", "n_sims": None, "host_fraction": None}
    return st.session_state.get("predictions")


def ensure_predictions():
    """Return predictions, or stop the page asking the user to simulate."""
    preds = get_predictions()
    if preds is None:
        st.warning(
            "No simulation results yet. Open the **Tournament Simulator** "
            "page and run a simulation first."
        )
        st.stop()
    return preds


def needs_finish_columns(preds):
    """Stop the page if the predictions predate the finishing-position update."""
    if "P(1st)" not in preds.columns:
        st.warning(
            "These results were produced before the finishing-position update. "
            "Open the **Tournament Simulator** and run a fresh simulation."
        )
        st.stop()


# ===========================================================================
# BRACKET
# ===========================================================================
def round_name(match_id):
    """Human-readable round for a knockout match id."""
    if 73 <= match_id <= 88:
        return "Round of 32"
    if 89 <= match_id <= 96:
        return "Round of 16"
    if 97 <= match_id <= 100:
        return "Quarter-final"
    if 101 <= match_id <= 102:
        return "Semi-final"
    if match_id == 103:
        return "Third-place play-off"
    return "Final"


def projected_bracket(predictions, model):
    """
    Build one deterministic 'most likely' bracket:
      * each group's slot 1 / slot 2 = the modal winner / runner-up,
      * the eight third-place slots = the modal third-placed teams that are
        most likely to qualify, placed via the official eligibility rules,
      * every knockout tie is won by the team the model favours.

    Returns {match_id: (team_a, team_b, winner)}. This is a single
    representative outcome, not an average - probabilities live in the
    simulation table.
    """
    slots = {}
    third_team, third_score = {}, {}
    for g in cfg.GROUPS:
        grp = predictions[predictions["group"] == g]
        by_first = grp.sort_values("P(1st)", ascending=False)
        first = by_first.iloc[0]["team"]
        rest = grp[grp["team"] != first].sort_values("P(2nd)", ascending=False)
        slots[f"1{g}"] = first
        slots[f"2{g}"] = rest.iloc[0]["team"]
        by_third = grp.sort_values("P(3rd)", ascending=False)
        third_team[g] = by_third.iloc[0]["team"]
        third_score[g] = float(by_third.iloc[0]["P(advance)"])

    top8 = sorted(third_score, key=third_score.get, reverse=True)[:8]
    assignment = cfg.assign_third_places(top8)
    third_slot = {mid: third_team[g] for mid, g in assignment.items()}

    winners, losers, bracket = {}, {}, {}

    def resolve(slot, mid):
        if slot.startswith("3:"):
            return third_slot[mid]
        if slot.startswith("W"):
            return winners[slot]
        if slot.startswith("L"):
            return losers[slot]
        return slots[slot]

    def favourite(a, b):
        p = model.outcome_probabilities(a, b, neutral=True)
        return a if p["home_win"] >= p["away_win"] else b

    def play(matches):
        for mid, slot_a, slot_b in matches:
            a, b = resolve(slot_a, mid), resolve(slot_b, mid)
            w = favourite(a, b)
            winners[f"W{mid}"] = w
            losers[f"L{mid}"] = b if w == a else a
            bracket[mid] = (a, b, w)

    play(cfg.ROUND_OF_32)
    play(cfg.ROUND_OF_16)
    play(cfg.QUARTER_FINALS)
    play(cfg.SEMI_FINALS)
    play([cfg.THIRD_PLACE])
    play([cfg.FINAL])
    return bracket


def team_path(bracket, team):
    """The selected team's matches in the projected bracket, in round order.

    The third-place play-off (match 103) is skipped - it is not part of a
    team's run toward the title.
    """
    path = []
    for mid in sorted(bracket):
        if mid == 103:
            continue
        a, b, w = bracket[mid]
        if team in (a, b):
            path.append({
                "round": round_name(mid),
                "opponent": b if team == a else a,
                "won": w == team,
            })
    return path


def render_bracket(bracket, highlight=None):
    """Print the projected bracket round by round, winners in bold."""
    rounds = ["Round of 32", "Round of 16", "Quarter-final",
              "Semi-final", "Final"]
    by_round = {r: [] for r in rounds}
    for mid in sorted(bracket):
        rn = round_name(mid)
        if rn in by_round:
            by_round[rn].append(bracket[mid])

    for rn in rounds:
        st.markdown(f"#### {rn}")
        for a, b, w in by_round[rn]:
            line = f"{a}  vs  {b}  ->  **{w}**"
            if highlight in (a, b):
                line += "   *(selected team)*"
            st.write(line)


# ===========================================================================
# PAGE CHROME
# ===========================================================================
def page_setup(title):
    """First call on every page: set config and render the status sidebar."""
    st.set_page_config(page_title=f"{title} - WC 2026 Dixon-Coles",
                       layout="wide")
    info = model_info()
    with st.sidebar:
        st.markdown("#### Model status")
        if info is None:
            st.error("Not fitted - run src/dixon_coles.py")
        else:
            st.caption(f"Fitted {str(info['fitted_at'])[:10]}  |  "
                       f"{info['n_teams']} teams")
            st.caption(f"home adv {info['home_advantage']:+.3f}  |  "
                       f"rho {info['rho']:+.3f}")
        meta = st.session_state.get("sim_meta")
        if meta:
            st.markdown("#### Current results")
            if meta.get("n_sims"):
                st.caption(f"{meta['n_sims']:,} sims  |  "
                           f"host adv {meta['host_fraction']}")
            else:
                st.caption(f"source: {meta['source']}")
