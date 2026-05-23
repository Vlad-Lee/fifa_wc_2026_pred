import io
import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np


INITIAL_RATING = 1500
K_FACTORS = {
    'FIFA World Cup': 60,
    'FIFA World Cup qualification': 40,
    'UEFA Euro': 50,
    'Copa América': 50,
    'AFC Asian Cup': 40,
    'Africa Cup of Nations': 40,
    'Friendly': 20,
}
DEFAULT_K = 30  # for any tournament not listed above


def get_k_factor(tournament):
    for key in K_FACTORS:
        if key in tournament:
            return K_FACTORS[key]
    return DEFAULT_K


def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_a, rating_b, result, k):
    """result: 1 = team A win, 0.5 = draw, 0 = team A loss"""
    exp = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (result - exp)
    new_b = rating_b + k * ((1 - result) - (1 - exp))
    return new_a, new_b


def compute_elo_ratings(df, initial_ratings=None):
    """
    Iterate through matches chronologically and compute
    rolling Elo ratings. Returns the match df with Elo
    columns added, plus a final ratings dict.
    """
    ratings = dict(initial_ratings) if initial_ratings else {}
    pre_elo_home, pre_elo_away = [], []

    df = df.sort_values('date').reset_index(drop=True)

    for _, row in df.iterrows():
        home, away = row['home_team'], row['away_team']

        if home not in ratings:
            ratings[home] = INITIAL_RATING
        if away not in ratings:
            ratings[away] = INITIAL_RATING

        pre_elo_home.append(ratings[home])
        pre_elo_away.append(ratings[away])

        if row['home_score'] > row['away_score']:
            result = 1.0
        elif row['home_score'] == row['away_score']:
            result = 0.5
        else:
            result = 0.0

        k = get_k_factor(row['tournament'])
        ratings[home], ratings[away] = update_elo(
            ratings[home], ratings[away], result, k
        )

    df['pre_match_elo_home'] = pre_elo_home
    df['pre_match_elo_away'] = pre_elo_away
    df['elo_diff'] = df['pre_match_elo_home'] - df['pre_match_elo_away']

    return df, ratings


def package_matrices(matrices, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, "modeling_matrices_package.zip")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, df in matrices.items():
            buf = io.StringIO()
            if isinstance(df, pd.Series):
                df = df.to_frame(name)
            df.to_csv(buf, index=False)
            z.writestr(f"{name}.csv", buf.getvalue())

    print(f"Saved modeling matrices to {zip_path}")


if __name__ == "__main__":
    base_dir     = Path(__file__).resolve().parent.parent
    zip_path     = base_dir / "data" / "processed" / "modeling" / "modeling_matrices_package.zip"
    out_dir      = base_dir / "data" / "processed" / "modeling"
    ratings_path = out_dir / "final_elo_ratings.json"

    # Load all matrices from zip
    with zipfile.ZipFile(zip_path) as z:
        with z.open("train.csv") as f:
            train = pd.read_csv(f, parse_dates=['date'])
        with z.open("test.csv") as f:
            test = pd.read_csv(f, parse_dates=['date'])
        with z.open("fixtures.csv") as f:
            fixtures = pd.read_csv(f, parse_dates=['date'])

    # Compute Elo — carry ratings from train into test
    train, ratings = compute_elo_ratings(train, initial_ratings=None)
    test,  ratings = compute_elo_ratings(test,  initial_ratings=ratings)

    # Rebuild zip with Elo-enriched train and test
    package_matrices({"train": train, "test": test, "fixtures": fixtures}, out_dir)

    # Save final ratings separately as JSON
    with open(ratings_path, 'w') as f:
        json.dump(ratings, f, indent=2)

    print(train[['date', 'home_team', 'away_team',
                 'pre_match_elo_home', 'pre_match_elo_away', 'elo_diff']].tail(10))

    print("\nTop 20 teams by final Elo:")
    top20 = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:20]
    for team, rating in top20:
        print(f"  {team:<30} {rating:.1f}")

    print(f"\nSaved final Elo ratings to {ratings_path}")