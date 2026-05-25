"""
simulate_wc2026.py
==================

Monte-Carlo simulation of the 2026 FIFA World Cup, driven by the fitted
Dixon-Coles model (dixon_coles.py) and the tournament structure
(wc2026_config.py).

HOW THE SIMULATION WORKS
------------------------
One "simulation" plays the entire tournament once:

  1. Group stage - every group is a round-robin (6 matches). Each match is
     played by drawing a scoreline from the Dixon-Coles distribution for that
     pairing. Host nations get a fraction (HOST_ADVANTAGE_FRACTION) of the
     model's home-advantage term in their group games; everything else is
     treated as a neutral venue.
  2. Group tables are ranked (points, then goal difference, then goals
     scored, then a random draw standing in for FIFA's remaining tie-breakers).
  3. The eight best third-placed teams are identified and slotted into the
     bracket using the official eligibility rules (see wc2026_config.py).
  4. The knockout bracket is played match by match. A knockout match that is
     level after 90 minutes goes to extra time, and if still level, to a
     penalty shootout - both simulated explicitly (see simulate_knockout).
  5. We record how far every team got.

Repeating this many thousands of times and averaging turns those individual
play-throughs into probabilities: P(win group), P(reach the quarter-finals),
P(win the tournament), and so on.

Run it:  python src/simulate_wc2026.py [n_simulations]
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import wc2026_config as cfg
from dixon_coles import DixonColesModel, MAX_GOALS


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
N_SIMULATIONS = 20_000      # default; override on the command line
RANDOM_SEED = 2026

# Penalty-shootout conversion probability per kick. Historical men's World Cup
# shootout kicks convert at roughly 70-75%. We use a single team-neutral value
# - the model does not carry team-specific penalty skill.
PENALTY_CONVERSION = 0.75

# Host advantage. The three host nations receive HOST_ADVANTAGE_FRACTION of
# the model's fitted home-advantage term in their GROUP-stage matches:
#   0.0 = no host advantage (every match fully neutral)
#   0.5 = half the historical home edge (current setting)
#   1.0 = the full home-advantage term
# Half is a deliberate compromise: a host genuinely benefits, but a 48-team
# tournament spread across a whole continent is a weaker home effect than a
# single-country World Cup. Knockout games are always neutral - knockout
# venues are fixed in advance, so a host playing there is just bracket luck.
# The fitted model still estimates home_advantage from historical home/away
# matches; this fraction controls only how much of it the simulation applies.
HOST_ADVANTAGE_FRACTION = 0.5

RNG = np.random.default_rng(RANDOM_SEED)


# ===========================================================================
# STEP 0 — RECONCILE TEAM NAMES BETWEEN THE DRAW AND THE MODEL
# ===========================================================================
def resolve_team_names(model):
    """
    Every team in the 2026 draw must exist in the fitted model (i.e. in
    results.csv). Returns {draw_name: model_name}. Raises a clear error if a
    team cannot be matched - fix it by adding an entry to cfg.NAME_ALIASES.
    """
    mapping, missing = {}, []
    for team in (t for grp in cfg.GROUPS.values() for t in grp):
        candidate = cfg.NAME_ALIASES.get(team, team)
        if candidate in model.attack:
            mapping[team] = candidate
        else:
            missing.append(team)
    if missing:
        raise KeyError(
            "These 2026 teams are not in the fitted model:\n  "
            + ", ".join(missing)
            + "\nAdd them to NAME_ALIASES in wc2026_config.py with the exact "
              "spelling used in results.csv."
        )
    return mapping


# ===========================================================================
# STEP 1 — SAMPLE A SCORELINE FROM THE MODEL
# ===========================================================================
# Building a Dixon-Coles score matrix costs a little maths, and the same
# pairings recur in every one of the thousands of simulations. So we cache the
# cumulative distribution for each (home, away, home_factor, rate) combination.
# Sampling is then just one uniform draw + a binary search, which is much
# faster than rebuilding the matrix every time.
_cdf_cache = {}


def _score_cdf(model, home, away, home_factor, rate_scale):
    key = (home, away, round(home_factor, 4), round(rate_scale, 4))
    cached = _cdf_cache.get(key)
    if cached is None:
        matrix = model.score_matrix(home, away, home_factor=home_factor,
                                    max_goals=MAX_GOALS, rate_scale=rate_scale)
        # Flatten row-major (index = home_goals * W + away_goals), then form
        # the cumulative distribution for inverse-transform sampling.
        cached = np.cumsum(matrix.ravel())
        _cdf_cache[key] = cached
    return cached


def sample_score(model, home, away, home_factor=0.0, rate_scale=1.0):
    """
    Draw one scoreline (home_goals, away_goals) from the Dixon-Coles joint
    distribution for this pairing, by inverse-transform sampling.

    `home_factor` = fraction of home advantage applied to `home` (0.0 = neutral
    venue, the default). `rate_scale` = 1.0 for a normal 90 minutes; 1/3 for a
    30-minute period of extra time (goals accrue at one third the full rate).
    """
    cdf = _score_cdf(model, home, away, home_factor, rate_scale)
    idx = int(np.searchsorted(cdf, RNG.random()))
    idx = min(idx, len(cdf) - 1)          # guard the floating-point top end
    home_goals, away_goals = divmod(idx, MAX_GOALS + 1)
    return int(home_goals), int(away_goals)


# ===========================================================================
# STEP 2 — PLAY ONE GROUP
# ===========================================================================
def simulate_group(model, names, hosts):
    """
    Play a single group (4 teams, round-robin of 6 matches) and return the
    four teams ranked 1st-4th, plus each team's record.

    `names` are model-space team names. `hosts` is the set of host nations.
    """
    table = {t: {"pts": 0, "gf": 0, "ga": 0} for t in names}

    # All 6 distinct pairings.
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            t1, t2 = names[i], names[j]

            # Decide the venue. At most one host can be in a group (the three
            # hosts are drawn into different groups), so if a host is present
            # it plays at home and receives HOST_ADVANTAGE_FRACTION of the
            # home-advantage term. Every other group game is neutral.
            if t1 in hosts:
                home, away, hf = t1, t2, HOST_ADVANTAGE_FRACTION
            elif t2 in hosts:
                home, away, hf = t2, t1, HOST_ADVANTAGE_FRACTION
            else:
                home, away, hf = t1, t2, 0.0

            hg, ag = sample_score(model, home, away, home_factor=hf)

            table[home]["gf"] += hg
            table[home]["ga"] += ag
            table[away]["gf"] += ag
            table[away]["ga"] += hg
            if hg > ag:
                table[home]["pts"] += 3
            elif hg < ag:
                table[away]["pts"] += 3
            else:
                table[home]["pts"] += 1
                table[away]["pts"] += 1

    ranked = rank_teams(table)
    return ranked, table


def rank_teams(table):
    """
    Rank teams best-to-worst by: points, then goal difference, then goals
    scored, then a random draw. The random draw stands in for FIFA's lower
    tie-breakers (disciplinary record, then drawing of lots).
    """
    def sort_key(team):
        rec = table[team]
        gd = rec["gf"] - rec["ga"]
        return (rec["pts"], gd, rec["gf"], RNG.random())

    return sorted(table, key=sort_key, reverse=True)


# ===========================================================================
# STEP 3 — A KNOCKOUT MATCH: 90 MIN -> EXTRA TIME -> PENALTIES
# ===========================================================================
def penalty_shootout(model, team_a, team_b):
    """
    Simulate a penalty shootout explicitly, kick by kick.

    Format: each team takes 5 kicks; if still level it goes to sudden death
    (one kick each) until one team scores and the other misses. Every kick is
    an independent success with probability PENALTY_CONVERSION.

    Note: we always simulate all 5 first-round kicks even after the result is
    mathematically decided. This does not change who wins - if a team's lead
    is already unassailable, taking the remaining kicks cannot overturn it -
    and it keeps the code simple.

    Returns the winning team.
    """
    a = int(np.sum(RNG.random(5) < PENALTY_CONVERSION))
    b = int(np.sum(RNG.random(5) < PENALTY_CONVERSION))

    # Sudden death: keep taking one kick each until they differ.
    while a == b:
        a += RNG.random() < PENALTY_CONVERSION
        b += RNG.random() < PENALTY_CONVERSION

    return team_a if a > b else team_b


def simulate_knockout(model, team_a, team_b):
    """
    Play one knockout match to a finish. Returns (winner, loser).

    Stage 1 - 90 minutes: draw a scoreline normally.
    Stage 2 - extra time: only if level. Two 15-minute halves = 30 minutes,
              so goals are drawn from the model at one third of the 90-minute
              rate (rate_scale = 1/3) and added to the aggregate score.
    Stage 3 - penalties: only if STILL level after extra time.

    Knockout matches are treated as neutral-venue (host countries share the
    knockout venues).
    """
    a, b = sample_score(model, team_a, team_b)               # neutral venue

    if a == b:
        # Extra time - an independent 30-minute mini-match, still neutral.
        ea, eb = sample_score(model, team_a, team_b,
                              rate_scale=1.0 / 3.0)
        a += ea
        b += eb

    if a > b:
        return team_a, team_b
    if b > a:
        return team_b, team_a

    # Still level after extra time -> penalties.
    winner = penalty_shootout(model, team_a, team_b)
    loser = team_b if winner == team_a else team_a
    return winner, loser


# ===========================================================================
# STEP 4 — PLAY THE WHOLE TOURNAMENT ONCE
# ===========================================================================
def simulate_tournament(model, groups, hosts):
    """
    Run one complete tournament. Returns {team: stage_reached}, where
    stage_reached is:
        1 = group stage only        4 = quarter-finalist
        2 = reached round of 32     5 = semi-finalist
        3 = reached round of 16     6 = finalist
                                    7 = champion
    Also returns the set of group winners, and {team: group finishing
    position 1-4} for the group-stage standings distribution.
    """
    # ---- group stage ----------------------------------------------------
    slots = {}                       # "1A"/"2A" -> team
    third_place = {}                 # group letter -> (team, pts, gd, gf)
    stage = {}                       # team -> furthest stage reached
    finish = {}                      # team -> group finishing position (1-4)
    group_winners = set()

    for letter, names in groups.items():
        ranked, table = simulate_group(model, names, hosts)
        slots[f"1{letter}"] = ranked[0]
        slots[f"2{letter}"] = ranked[1]
        group_winners.add(ranked[0])

        for position, team in enumerate(ranked, start=1):
            finish[team] = position              # 1st / 2nd / 3rd / 4th in group
        for team in names:
            stage[team] = 1                      # everyone: group stage
        for team in ranked[:2]:
            stage[team] = 2                      # top two reach the round of 32

        third = ranked[2]
        rec = table[third]
        third_place[letter] = (third, rec["pts"],
                               rec["gf"] - rec["ga"], rec["gf"])

    # ---- rank the twelve third-placed teams, keep the best eight --------
    third_ranked = sorted(
        third_place.items(),
        key=lambda kv: (kv[1][1], kv[1][2], kv[1][3], RNG.random()),
        reverse=True,
    )
    qualifying_groups = [letter for letter, _ in third_ranked[:8]]
    for letter in qualifying_groups:
        team = third_place[letter][0]
        stage[team] = 2                          # this third-placed team advances

    # Slot those eight groups into the bracket via the official eligibility
    # rules, then expose each as a "3<group>" slot the bracket can look up.
    assignment = cfg.assign_third_places(qualifying_groups)  # {match_id: group}
    third_slot_team = {}                         # match_id -> actual team
    for match_id, group_letter in assignment.items():
        third_slot_team[match_id] = third_place[group_letter][0]

    # ---- knockout bracket ----------------------------------------------
    winners = {}                                 # "W73" -> team
    losers = {}                                  # "L101" -> team (3rd-place game)

    def resolve(slot, match_id):
        """Turn a bracket slot code into an actual team."""
        if slot.startswith("3:"):
            return third_slot_team[match_id]      # third-place slot
        if slot.startswith("W"):
            return winners[slot]                  # winner of an earlier match
        if slot.startswith("L"):
            return losers[slot]                   # loser of an earlier match
        return slots[slot]                        # "1A" / "2B" group slot

    # Round of 32 -> ... -> Final, in order. The stage value reached by the
    # WINNER of each round is one higher than the round it just won.
    rounds = [
        (cfg.ROUND_OF_32, 3),     # win a R32 match  -> reached round of 16
        (cfg.ROUND_OF_16, 4),     # win a R16 match  -> reached quarter-finals
        (cfg.QUARTER_FINALS, 5),  # win a QF         -> reached semi-finals
        (cfg.SEMI_FINALS, 6),     # win a SF         -> reached the final
    ]
    for matches, winner_stage in rounds:
        for match_id, slot_a, slot_b in matches:
            team_a = resolve(slot_a, match_id)
            team_b = resolve(slot_b, match_id)
            w, l = simulate_knockout(model, team_a, team_b)
            winners[f"W{match_id}"] = w
            losers[f"L{match_id}"] = l
            stage[w] = max(stage.get(w, 0), winner_stage)

    # Third-place play-off (does not change anyone's recorded stage).
    mid, sa, sb = cfg.THIRD_PLACE
    simulate_knockout(model, resolve(sa, mid), resolve(sb, mid))

    # Final.
    mid, sa, sb = cfg.FINAL
    champion, _ = simulate_knockout(model, resolve(sa, mid), resolve(sb, mid))
    stage[champion] = 7

    return stage, group_winners, finish


# ===========================================================================
# STEP 5 — RUN MANY SIMULATIONS AND AGGREGATE
# ===========================================================================
def run(model, n_simulations):
    """Run the tournament n_simulations times and return a results table."""
    groups = {                       # groups in model-space team names
        letter: [NAME_MAP[t] for t in teams]
        for letter, teams in cfg.GROUPS.items()
    }
    hosts = {NAME_MAP[t] for t in cfg.HOSTS}
    team_group = {NAME_MAP[t]: letter
                  for letter, teams in cfg.GROUPS.items() for t in teams}

    # Counters: for each team, how many sims it reached at least each stage.
    reached = defaultdict(lambda: np.zeros(8, dtype=np.int64))  # index = stage
    group_wins = defaultdict(int)
    finish_counts = defaultdict(lambda: np.zeros(4, dtype=np.int64))  # pos 1-4

    for s in range(n_simulations):
        stage, winners, finish = simulate_tournament(model, groups, hosts)
        for team, st in stage.items():
            # A team that reached stage `st` also reached every lower stage.
            reached[team][:st + 1] += 1
        for team in winners:
            group_wins[team] += 1
        for team, position in finish.items():
            finish_counts[team][position - 1] += 1

        if (s + 1) % max(1, n_simulations // 10) == 0:
            print(f"  {s + 1:,} / {n_simulations:,} simulations done")

    # Build the output table. Stage codes:
    #   2 reach R32, 3 reach R16, 4 reach QF, 5 reach SF, 6 reach final, 7 win.
    rows = []
    for team in sorted(team_group):
        r = reached[team]
        fc = finish_counts[team]
        rows.append({
            "team": team,
            "group": team_group[team],
            "P(1st)": fc[0] / n_simulations,
            "P(2nd)": fc[1] / n_simulations,
            "P(3rd)": fc[2] / n_simulations,
            "P(4th)": fc[3] / n_simulations,
            "P(win group)": group_wins[team] / n_simulations,
            "P(advance)": r[2] / n_simulations,        # out of the group
            "P(round of 16)": r[3] / n_simulations,
            "P(quarter-final)": r[4] / n_simulations,
            "P(semi-final)": r[5] / n_simulations,
            "P(final)": r[6] / n_simulations,
            "P(champion)": r[7] / n_simulations,
        })

    df = pd.DataFrame(rows).sort_values("P(champion)", ascending=False)
    return df.reset_index(drop=True)


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else N_SIMULATIONS

    base_dir = Path(__file__).resolve().parent.parent
    params_path = base_dir / "data" / "processed" / "modeling" / \
        "dixon_coles_params.json"
    out_path = base_dir / "data" / "processed" / "modeling" / \
        "wc2026_predictions.csv"

    print("Loading fitted Dixon-Coles model ...")
    model = DixonColesModel.load(params_path)

    print("Checking that every 2026 team is in the model ...")
    NAME_MAP = resolve_team_names(model)        # draw name -> model name
    print(f"  all {len(NAME_MAP)} teams matched")

    print(f"Running {n_sims:,} tournament simulations ...")
    predictions = run(model, n_sims)

    predictions.to_csv(out_path, index=False)
    print(f"\nSaved full prediction table to {out_path}")

    # Pretty console summary.
    pd.set_option("display.float_format", lambda v: f"{v:6.1%}")
    pd.set_option("display.max_rows", None)
    print("\n" + "=" * 78)
    print(f"2026 WORLD CUP — DIXON-COLES FORECAST ({n_sims:,} simulations)")
    print("=" * 78)
    show = ["team", "group", "P(advance)", "P(quarter-final)",
            "P(semi-final)", "P(final)", "P(champion)"]
    print(predictions[show].head(24).to_string(index=False))
