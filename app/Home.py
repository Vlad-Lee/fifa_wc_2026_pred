"""
Home.py - entry point of the 2026 World Cup Dixon-Coles Streamlit app.

Run it with:  streamlit run app/Home.py
"""

import streamlit as st

import lib

lib.page_setup("Home")

st.title("2026 World Cup - Dixon-Coles forecast")

st.markdown(
    "This app forecasts the 2026 FIFA World Cup with a **Dixon-Coles** goal "
    "model. Each team has an attacking and a defensive rating; every match is "
    "two correlated Poisson goal counts; recent results count more than old "
    "ones. The tournament is then played out thousands of times by "
    "Monte-Carlo simulation to turn those match probabilities into "
    "advancement and title odds."
)

info = lib.model_info()
if info is None:
    st.error(
        "The model has not been fitted yet. Run `python src/dixon_coles.py` "
        "to create `dixon_coles_params.json`, then reload this page."
    )
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Teams in the model", info["n_teams"])
    c2.metric("Home advantage", f"{info['home_advantage']:+.3f}")
    c3.metric("Low-score rho", f"{info['rho']:+.3f}")
    c4.metric("Fitted", str(info["fitted_at"])[:10])

st.markdown("### Pages")
st.markdown(
    "- **Tournament Simulator** - run N simulations; full probability table, "
    "sortable by any column.\n"
    "- **Head-to-Head** - any two teams: scoreline heatmap and win/draw/loss.\n"
    "- **Bracket Explorer** - a team's road through the knockouts, or the "
    "most-likely bracket.\n"
    "- **Group Stage** - each group's finishing-position distribution.\n"
    "- **Team Ratings** - the fitted attack/defense numbers.\n"
    "- **Methodology** - how the model and simulation work.\n"
    "- **Most-Likely Bracket** - the single most probable bracket end to end."
)

st.markdown("### Key assumptions")
st.markdown(
    "- World Cup matches are played at neutral venues; the three host "
    "nations get half the model's home advantage in their group games.\n"
    "- Knockout ties level after 90 minutes go to extra time, and if still "
    "level to an explicit penalty shootout.\n"
    "- The model is fitted only on international match scores - it carries no "
    "injury, squad-selection or pre-tournament form information."
)

st.divider()
if info is not None:
    st.caption(
        "Start on the Tournament Simulator page to generate the probabilities "
        "that the Bracket Explorer and Group Stage pages read."
    )
