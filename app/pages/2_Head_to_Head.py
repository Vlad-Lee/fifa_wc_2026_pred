"""Head-to-Head - pick any two teams and see the modelled match outcome."""

import altair as alt
import pandas as pd
import streamlit as st

import lib

lib.page_setup("Head-to-Head")
st.title("Head-to-Head")
st.caption("Pick any two teams for a single modelled match. This uses the "
           "Dixon-Coles model directly.")

model = lib.ensure_model()
teams = sorted(model.attack)


def default_index(name, fallback):
    return teams.index(name) if name in teams else fallback


c1, c2, c3 = st.columns([2, 2, 2])
team_a = c1.selectbox("Team A", teams, index=default_index("Spain", 0))
team_b = c2.selectbox("Team B", teams, index=default_index("France", 1))
venue = c3.radio("Venue", ["Neutral (World Cup)", "Team A at home"], index=0)

if team_a == team_b:
    st.warning("Pick two different teams.")
    st.stop()

neutral = venue.startswith("Neutral")
lam, mu = model.expected_goals(team_a, team_b, neutral=neutral)
probs = model.outcome_probabilities(team_a, team_b, neutral=neutral)

m1, m2, m3 = st.columns(3)
m1.metric(f"{team_a} win", f"{probs['home_win']:.1%}")
m2.metric("Draw", f"{probs['draw']:.1%}")
m3.metric(f"{team_b} win", f"{probs['away_win']:.1%}")
st.caption(f"Expected goals - {team_a}: {lam:.2f}   |   {team_b}: {mu:.2f}")

# --- scoreline heatmap ----------------------------------------------------
matrix = model.score_matrix(team_a, team_b, neutral=neutral, max_goals=6)
cells = [
    {"a_goals": a, "b_goals": b, "p": float(matrix[a, b])}
    for a in range(matrix.shape[0])
    for b in range(matrix.shape[1])
]
grid = pd.DataFrame(cells)

st.markdown("### Scoreline probabilities")
heat = (
    alt.Chart(grid)
    .mark_rect()
    .encode(
        x=alt.X("b_goals:O", title=f"{team_b} goals"),
        y=alt.Y("a_goals:O", title=f"{team_a} goals", sort="descending"),
        color=alt.Color("p:Q", title="probability",
                        scale=alt.Scale(scheme="blues")),
        tooltip=[alt.Tooltip("a_goals:O", title=f"{team_a}"),
                 alt.Tooltip("b_goals:O", title=f"{team_b}"),
                 alt.Tooltip("p:Q", format=".2%", title="probability")],
    )
)
st.altair_chart(heat, use_container_width=True)

st.markdown("### Most likely scorelines")
for _, r in grid.sort_values("p", ascending=False).head(6).iterrows():
    st.write(f"{team_a} {int(r.a_goals)} - {int(r.b_goals)} {team_b}"
             f"  ({r.p:.1%})")
