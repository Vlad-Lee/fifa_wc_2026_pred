"""Bracket Explorer - a team's road through the knockouts, or the projected
bracket as a whole."""

import altair as alt
import pandas as pd
import streamlit as st

import lib

lib.page_setup("Bracket Explorer")
st.title("Bracket Explorer")

model = lib.ensure_model()
preds = lib.ensure_predictions()
lib.needs_finish_columns(preds)

view = st.radio(
    "View",
    ["Selected team's path", "Most-likely bracket"],
    horizontal=True,
)

# ===========================================================================
# VIEW 1 - one team's per-round probabilities + its projected path
# ===========================================================================
if view == "Selected team's path":
    teams = sorted(preds["team"])
    team = st.selectbox("Team", teams)
    row = preds[preds["team"] == team].iloc[0]

    stages = [
        ("Reach knockouts", "P(advance)"),
        ("Round of 16", "P(round of 16)"),
        ("Quarter-final", "P(quarter-final)"),
        ("Semi-final", "P(semi-final)"),
        ("Final", "P(final)"),
        ("Champion", "P(champion)"),
    ]
    funnel = pd.DataFrame({
        "stage": [name for name, _ in stages],
        "p": [float(row[col]) for _, col in stages],
    })

    st.markdown(f"### {team} - how far it goes")
    chart = (
        alt.Chart(funnel)
        .mark_bar()
        .encode(
            x=alt.X("p:Q", axis=alt.Axis(format="%"), title="probability", scale=alt.Scale(domain=[0,1])),
            y=alt.Y("stage:N", sort=[s for s, _ in stages], title=None),
            tooltip=[alt.Tooltip("p:Q", format=".1%")],
        )
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("### Projected path")
    st.caption("The opponents this team would meet in the single most-likely "
               "bracket. The probabilities above are exact; this path is one "
               "representative scenario.")
    bracket = lib.projected_bracket(preds, model)
    path = lib.team_path(bracket, team)
    if not path:
        st.info(f"{team} does not reach the knockout stage in the "
                "most-likely bracket.")
    else:
        for step in path:
            result = "advances" if step["won"] else "eliminated"
            st.write(f"**{step['round']}**  vs  {step['opponent']}  ->  "
                     f"{result}")

# ===========================================================================
# VIEW 2 - the whole projected bracket
# ===========================================================================
else:
    st.markdown("### Most-likely bracket")
    st.caption("Every slot filled with the single most likely team, each tie "
               "won by the model's favourite. One representative outcome - "
               "not an average.")
    bracket = lib.projected_bracket(preds, model)
    champion = bracket[104][2]
    st.success(f"Projected champion: {champion}")
    lib.render_bracket(bracket)
