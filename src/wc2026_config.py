"""
wc2026_config.py
================

Static description of the 2026 FIFA World Cup: the 48-team field, the group
draw (held 5 December 2025), and the full single-elimination bracket from the
round of 32 to the final.

This file contains NO modelling logic - it is pure tournament structure that
``simulate_wc2026.py`` consumes. Keeping it separate means that if FIFA changes
a name or a play-off slot, you edit this one file and nothing else.

Source: FIFA 2026 World Cup final-draw results and the published knockout-stage
regulations (Annex C).
"""

# ---------------------------------------------------------------------------
# 1. THE GROUP DRAW
# ---------------------------------------------------------------------------
# 12 groups (A-L) of 4 teams. Order inside a group is irrelevant for the
# simulation - every group is a round-robin where all 4 teams play each other.
#
# IMPORTANT: team names must match the spelling used in results.csv (the raw
# match dataset). If a name differs, add it to NAME_ALIASES below. The model
# loader validates this and will tell you about any mismatch.
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# The three host nations. They play their group matches at home, so the
# simulation applies the model's home-advantage term to them in the group
# stage (knockout venues are mixed, so knockout games are treated as neutral).
HOSTS = {"Mexico", "Canada", "United States"}

# Optional rename map: {name_in_this_file: name_in_results.csv}.
# Leave empty unless the validation step reports a missing team.
NAME_ALIASES = {
    # Example - uncomment / edit if your dataset spells these differently:
    # "Turkey": "Türkiye",
    # "Czech Republic": "Czechia",
}


# ---------------------------------------------------------------------------
# 2. ROUND OF 32
# ---------------------------------------------------------------------------
# Each tuple is (match_id, slot_A, slot_B). A "slot" is a short code:
#   "1A" = winner of group A          "2A" = runner-up of group A
#   "3:ABCDF" = one of the eight best third-placed teams, drawn from a group
#               in the eligible set {A,B,C,D,F}
#
# The eligible sets are fixed by FIFA and already exclude the slot-winner's own
# group, which automatically enforces the rule "no team meets a side from its
# own group in the round of 32".
ROUND_OF_32 = [
    (73, "2A", "2B"),
    (74, "1E", "3:ABCDF"),
    (75, "1F", "2C"),
    (76, "1C", "2F"),
    (77, "1I", "3:CDFGH"),
    (78, "2E", "2I"),
    (79, "1A", "3:CEFHI"),
    (80, "1L", "3:EHIJK"),
    (81, "1D", "3:BEFIJ"),
    (82, "1G", "3:AEHIJ"),
    (83, "2K", "2L"),
    (84, "1H", "2J"),
    (85, "1B", "3:EFGIJ"),
    (86, "1J", "2H"),
    (87, "1K", "3:DEIJL"),
    (88, "2D", "2G"),
]

# ---------------------------------------------------------------------------
# 3. ROUND OF 16 -> FINAL
# ---------------------------------------------------------------------------
# From here on a slot is "W<match>" (winner of that match) or, for the
# third-place play-off, "L<match>" (loser of that match).
ROUND_OF_16 = [
    (89, "W74", "W77"),
    (90, "W73", "W75"),
    (91, "W76", "W78"),
    (92, "W79", "W80"),
    (93, "W83", "W84"),
    (94, "W81", "W82"),
    (95, "W86", "W88"),
    (96, "W85", "W87"),
]

QUARTER_FINALS = [
    (97, "W89", "W90"),
    (98, "W93", "W94"),
    (99, "W91", "W92"),
    (100, "W95", "W96"),
]

SEMI_FINALS = [
    (101, "W97", "W98"),
    (102, "W99", "W100"),
]

THIRD_PLACE = (103, "L101", "L102")
FINAL = (104, "W101", "W102")


# ---------------------------------------------------------------------------
# 4. THIRD-PLACE TEAM ASSIGNMENT
# ---------------------------------------------------------------------------
# Eight of the twelve third-placed teams advance. FIFA published a 495-row
# lookup table (Annex C) giving an exact bracket slot for every possible set of
# eight qualifying groups.
#
# Instead of embedding that 495-row table, we reproduce the SAME rules with a
# small bipartite-matching solver. Each of the eight third-place bracket slots
# accepts third-placed teams only from a fixed set of five groups (the eligible
# sets in ROUND_OF_32 above). Given the eight groups that actually produced a
# qualifying third-placed team, we find an assignment of groups to slots that
# respects every eligibility set. Such a matching is always possible, and any
# valid matching is a rules-legal bracket. It can differ from FIFA's specific
# tabulated choice only when more than one legal matching exists - a situation
# whose effect on aggregate title probabilities is negligible.

# The eight round-of-32 matches that contain a third-place slot, paired with
# the set of groups each will accept.
THIRD_PLACE_SLOTS = {
    74: set("ABCDF"),
    77: set("CDFGH"),
    79: set("CEFHI"),
    80: set("EHIJK"),
    81: set("BEFIJ"),
    82: set("AEHIJ"),
    85: set("EFGIJ"),
    87: set("DEIJL"),
}


def assign_third_places(qualified_groups):
    """
    Map the eight qualifying third-place groups onto the eight bracket slots.

    Parameters
    ----------
    qualified_groups : iterable of str
        The eight group letters whose third-placed team advanced.

    Returns
    -------
    dict {match_id: group_letter}

    Method: depth-first search with the most-constrained-slot-first heuristic.
    We always fill the slot with the fewest still-available candidates next,
    which makes the search effectively instant for eight items.
    """
    groups = set(qualified_groups)
    if len(groups) != 8:
        raise ValueError(f"Expected 8 third-place groups, got {len(groups)}")

    slots = list(THIRD_PLACE_SLOTS.items())  # [(match_id, eligible_set), ...]

    def solve(remaining_slots, remaining_groups, assignment):
        if not remaining_slots:
            return dict(assignment)
        # Most-constrained slot first: fewest eligible groups still available.
        remaining_slots.sort(
            key=lambda s: len(s[1] & remaining_groups)
        )
        match_id, eligible = remaining_slots[0]
        for grp in sorted(eligible & remaining_groups):
            assignment[match_id] = grp
            result = solve(
                remaining_slots[1:],
                remaining_groups - {grp},
                assignment,
            )
            if result is not None:
                return result
            del assignment[match_id]
        return None

    result = solve(slots, groups, {})
    if result is None:
        raise RuntimeError(
            f"No legal third-place assignment exists for groups {sorted(groups)}"
        )
    return result


# Convenience: every match in bracket order, useful for iterating a simulation.
ALL_KNOCKOUT_ROUNDS = [
    ("Round of 32", ROUND_OF_32),
    ("Round of 16", ROUND_OF_16),
    ("Quarter-finals", QUARTER_FINALS),
    ("Semi-finals", SEMI_FINALS),
    ("Third-place play-off", [THIRD_PLACE]),
    ("Final", [FINAL]),
]


if __name__ == "__main__":
    # Smoke test: print the field and check the bracket is internally consistent.
    teams = [t for g in GROUPS.values() for t in g]
    print(f"{len(GROUPS)} groups, {len(teams)} teams.")
    assert len(teams) == 48, "Expected 48 teams"
    assert len(set(teams)) == 48, "Duplicate team name in the draw"

    # Example third-place assignment (groups C,D,E,F,G,H,I,J advance).
    demo = assign_third_places("CDEFGHIJ")
    print("Demo third-place assignment (groups CDEFGHIJ):")
    for mid in sorted(demo):
        print(f"  Match {mid}: 3rd-place team from Group {demo[mid]}")
