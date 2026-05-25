"""Methodology - plain-language explanation of the model and simulation."""

import streamlit as st

import lib

lib.page_setup("Methodology")
st.title("Methodology")

st.markdown(
    "### The Dixon-Coles model\n"
    "Each match is modelled as two goal counts. The home team's goals follow "
    "a Poisson distribution with mean lambda, the away team's with mean mu, "
    "where\n"
    "```\n"
    "log(lambda) = attack(home) + defence(away) + home_advantage\n"
    "log(mu)     = attack(away) + defence(home)\n"
    "```\n"
    "Every team carries an **attack** and a **defense** rating. A neutral-"
    "venue match - the World Cup norm - simply drops the home-advantage term."
)

st.markdown(
    "### The low-score correction (rho)\n"
    "Two independent Poisson distributions misprice the very lowest scores - "
    "in particular they understate draws. Dixon and Coles multiply the "
    "probabilities of the 0-0, 1-0, 0-1 and 1-1 scorelines by a correction "
    "controlled by a single parameter, rho. A negative rho (the usual case) "
    "shifts probability toward 0-0 and 1-1, matching real football."
)

st.markdown(
    "### Time-decay weighting\n"
    "When the model is fitted, each historical match is weighted by "
    "`exp(-xi * age_in_days)`. A larger xi makes recent results count for "
    "more. The default corresponds to roughly a one-year half-life, so a "
    "match from five years ago carries far less weight than one from last "
    "month. All parameters are then estimated by maximum likelihood."
)

st.markdown(
    "### The tournament simulation\n"
    "One simulation plays the whole event: 12 group round-robins, ranked by "
    "points then goal difference then goals scored; the eight best third-"
    "placed teams; then the knockout bracket. A knockout tie level after 90 "
    "minutes goes to extra time (an independent 30-minute period at one third "
    "the goal rate) and, if still level, to an explicit penalty shootout. "
    "Running this many thousands of times turns single match probabilities "
    "into advancement and title probabilities."
)

st.markdown(
    "### Assumptions and limitations\n"
    "- World Cup games are treated as neutral-venue; the three hosts get half "
    "the home-advantage term in their group games only.\n"
    "- The model sees only historical match scores - no injuries, squad "
    "selection, or pre-tournament form.\n"
    "- Penalty shootouts use a single team-neutral conversion rate.\n"
    "- Third-placed teams are slotted into the bracket with a solver that "
    "respects FIFA's official eligibility rules; when several legal "
    "placements exist it may pick a different one than FIFA's table, with a "
    "negligible effect on aggregate probabilities."
)

info = lib.model_info()
if info:
    st.divider()
    st.caption(
        f"This model was fitted on {str(info['fitted_at'])[:10]} with "
        f"xi = {info['xi']}, covering {info['n_teams']} teams. "
        f"Home advantage = {info['home_advantage']:+.3f}, "
        f"rho = {info['rho']:+.3f}."
    )
