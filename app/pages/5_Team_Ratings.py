"""Team Ratings - the fitted Dixon-Coles attack/defence parameters."""

import altair as alt
import pandas as pd
import streamlit as st

import lib

lib.page_setup("Team Ratings")
st.title("Team Ratings")
st.caption("Every team's fitted attacking and defensive parameters. Higher "
           "attack is better; a more negative defence is better (it suppresses "
           "the opponent). Overall strength = attack minus defense.")

model = lib.ensure_model()

c1, c2 = st.columns(2)
c1.metric("Home advantage", f"{model.home_advantage:+.3f}",
          help="Added to the home team's log expected goals.")
c2.metric("Low-score rho", f"{model.rho:+.3f}",
          help="The Dixon-Coles correction; negative means draws upweighted.")

ratings = (
    pd.DataFrame([
        {"team": t,
         "attack": model.attack[t],
         "defence": model.defence[t],
         "strength": model.attack[t] - model.defence[t]}
        for t in model.attack
    ])
    .sort_values("strength", ascending=False)
    .reset_index(drop=True)
)
ratings.insert(0, "rank", ratings.index + 1)

only_wc = st.checkbox("Show only the 48 World Cup teams", value=True)
view = ratings
if only_wc:
    wc = {lib.cfg.NAME_ALIASES.get(t, t)
          for grp in lib.cfg.GROUPS.values() for t in grp}
    view = ratings[ratings["team"].isin(wc)].copy()
    view["rank"] = range(1, len(view) + 1)

num = {c: st.column_config.NumberColumn(c, format="%.3f")
       for c in ["attack", "defence", "strength"]}
st.dataframe(view, use_container_width=True, hide_index=True,
             column_config=num)

st.markdown("### Strongest teams")
top = view.nlargest(25, "strength")
chart = (
    alt.Chart(top)
    .mark_bar()
    .encode(
        x=alt.X("strength:Q", title="strength (attack - defence)"),
        y=alt.Y("team:N", sort="-x", title=None),
        tooltip=["team", alt.Tooltip("attack:Q", format=".3f"),
                 alt.Tooltip("defence:Q", format=".3f"),
                 alt.Tooltip("strength:Q", format=".3f")],
    )
)
st.altair_chart(chart, use_container_width=True)
