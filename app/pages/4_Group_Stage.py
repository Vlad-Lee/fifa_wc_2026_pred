"""Group Stage - each group's finishing-position probability distribution."""

import altair as alt
import streamlit as st

import lib

lib.page_setup("Group Stage")
st.title("Group Stage")
st.caption("How each group is likely to finish: the probability every team "
           "ends 1st, 2nd, 3rd or 4th.")

preds = lib.ensure_predictions()
lib.needs_finish_columns(preds)

group = st.selectbox("Group", sorted(lib.cfg.GROUPS.keys()))
g = preds[preds["group"] == group].copy()

# --- stacked finishing-position bar --------------------------------------
finish_cols = ["P(1st)", "P(2nd)", "P(3rd)", "P(4th)"]
long = g.melt(id_vars="team", value_vars=finish_cols,
              var_name="position", value_name="p")
team_order = g.sort_values("P(1st)", ascending=False)["team"].tolist()

st.markdown(f"### Group {group} - finishing-position distribution")
chart = (
    alt.Chart(long)
    .mark_bar()
    .encode(
        x=alt.X("p:Q", stack="normalize", axis=alt.Axis(format="%"),
                title="probability"),
        y=alt.Y("team:N", title=None, sort=team_order),
        color=alt.Color("position:N", title="finish",
                        scale=alt.Scale(scheme="blueorange")),
        order=alt.Order("position:N"),
        tooltip=["team", "position", alt.Tooltip("p:Q", format=".1%")],
    )
)
st.altair_chart(chart, use_container_width=True)

# --- table ----------------------------------------------------------------
st.markdown("### Probabilities")
cols = finish_cols + ["P(advance)"]
table = g[["team"] + cols].copy().sort_values("P(1st)", ascending=False)
table[cols] = (table[cols] * 100).round(1)
st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={c: st.column_config.NumberColumn(c, format="%.1f%%")
                   for c in cols},
)
st.caption("P(advance) includes reaching the round of 32 as one of the eight "
           "best third-placed teams, so it can exceed P(1st) + P(2nd).")
