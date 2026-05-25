"""Most-Likely Bracket - the single most probable bracket, end to end."""

import streamlit as st

import lib

lib.page_setup("Most-Likely Bracket")
st.title("Most-Likely Bracket")
st.caption("Every bracket slot filled with the single most likely team, and "
           "each tie won by the team the model favours. This is one "
           "representative outcome - not an average. For probabilities, use "
           "the Tournament Simulator and Bracket Explorer pages.")

model = lib.ensure_model()
preds = lib.ensure_predictions()
lib.needs_finish_columns(preds)

bracket = lib.projected_bracket(preds, model)

champion = bracket[104][2]
final_a, final_b, _ = bracket[104]
third = bracket[103][2]

c1, c2 = st.columns(2)
c1.metric("Projected champion", champion)
c2.metric("Projected final", f"{final_a} vs {final_b}")
st.caption(f"Projected third place: {third}")

st.divider()
lib.render_bracket(bracket)
