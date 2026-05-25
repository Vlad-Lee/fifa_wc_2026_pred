"""Tournament Simulator - run N Monte-Carlo simulations of the whole event."""

import altair as alt
import streamlit as st

import lib

lib.page_setup("Tournament Simulator")
st.title("Tournament Simulator")
st.caption("Run N Monte-Carlo simulations of the full tournament and read off "
           "every team's advancement and title probabilities.")

lib.ensure_model()

# --- controls -------------------------------------------------------------
c1, c2 = st.columns(2)
n_sims = c1.select_slider(
    "Number of simulations",
    options=[1_000, 2_000, 5_000, 10_000, 20_000, 50_000],
    value=5_000,
    help="More simulations = smoother probabilities but a longer wait.",
)
host = c2.slider(
    "Host advantage fraction (group games)", 0.0, 1.0, 0.5, 0.1,
    help="Share of the model's home advantage given to Mexico, Canada and "
         "the USA in their group matches.",
)

if st.button("Run simulation", type="primary"):
    try:
        with st.spinner(f"Simulating {n_sims:,} tournaments ..."):
            df = lib.run_simulation(n_sims, host)
    except KeyError as exc:
        st.error(
            f"A 2026 team is not in the fitted model: {exc}. Add it to "
            "`NAME_ALIASES` in `src/wc2026_config.py` (matching the spelling "
            "in results.csv) and reload."
        )
        st.stop()
    st.session_state["predictions"] = df
    st.session_state["sim_meta"] = {
        "source": "live run", "n_sims": n_sims, "host_fraction": host}
    st.success(
        f"Done - {n_sims:,} simulations. The result is cached and persisted "
        "to disk, so re-running these exact settings is instant, and the "
        "other pages now use these numbers. Use the button below to export a "
        "copy if you want one."
    )

# --- results --------------------------------------------------------------
preds = lib.get_predictions()
if preds is None:
    st.info("No results yet. Choose your settings and click **Run simulation**.")
    st.stop()

meta = st.session_state.get("sim_meta", {})
label = meta.get("source", "unknown")
if meta.get("n_sims"):
    label += f"  -  {meta['n_sims']:,} simulations, host advantage {meta['host_fraction']}"
st.markdown(f"**Showing:** {label}")

prob_cols = [c for c in preds.columns if c.startswith("P(")]
table = preds.copy()
table[prob_cols] = (table[prob_cols] * 100).round(1)

st.markdown("### Probability table")
st.caption("Click any column header to sort.")
st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={c: st.column_config.NumberColumn(c, format="%.1f%%")
                   for c in prob_cols},
)

st.markdown("### Title odds")
top = preds.nlargest(15, "P(champion)")
chart = (
    alt.Chart(top)
    .mark_bar()
    .encode(
        x=alt.X("P(champion):Q", axis=alt.Axis(format="%"), title="P(champion)"),
        y=alt.Y("team:N", sort="-x", title=None),
        tooltip=["team", alt.Tooltip("P(champion):Q", format=".1%")],
    )
)
st.altair_chart(chart, use_container_width=True)

st.download_button(
    "Download the full table (CSV)",
    preds.to_csv(index=False).encode("utf-8"),
    file_name="wc2026_predictions.csv",
    mime="text/csv",
)
