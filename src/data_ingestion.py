import os
from pathlib import Path

import duckdb
import pandas as pd


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
R2_BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"

TABLES = [
    "player_valuations",
    "players",
    "national_teams",
]


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_transfermarkt(tables=TABLES, base_url=R2_BASE):
    """
    Pull each table from the hosted .csv.gz files via DuckDB httpfs.
    No local DuckDB file is created — data is returned as DataFrames.
    """
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    data = {}
    for name in tables:
        url = f"{base_url}{name}.csv.gz"
        print(f"Fetching {name}...")
        data[name] = con.execute(
            f"SELECT * FROM read_csv_auto('{url}')"
        ).df()
        print(f"  → {len(data[name]):,} rows")

    con.close()
    return data


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_raw(data, out_dir):
    """Write each DataFrame to out_dir as a CSV."""
    os.makedirs(out_dir, exist_ok=True)
    for name, df in data.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {name}.csv → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    out_dir  = base_dir / "data" / "raw" / "transfermarkt"

    data = fetch_transfermarkt()
    save_raw(data, out_dir)

    print("\nDone. Files written to:", out_dir)