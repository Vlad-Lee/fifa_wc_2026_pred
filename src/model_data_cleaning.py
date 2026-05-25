import io
import os
import zipfile
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Non-FIFA territories and micronations to exclude
# ---------------------------------------------------------------------------
EXCLUDE_TEAMS = {
    'Canary Islands', 'Délvidék', 'Elba Island', 'Găgăuzia', 'Madrid',
    'Mapuche', 'Niue', 'Palau', 'Republic of St. Pauli', 'Ryūkyū',
    'Saugeais', 'Seborga', 'Silesia', 'West Papua', 'Yoruba Nation',
    'Ambazonia', 'Asturias', 'Cilento', 'Crimea', 'Maule Sur',
    'South Yemen', 'Surrey', 'Two Sicilies'
}

SPLIT_DATE = '2022-01-01'


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def load_raw_data(file_path):
    df = pd.read_csv(Path(file_path), low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df['neutral'] = df['neutral'].astype(bool)
    return df


def clean_data(df):
    """Remove non-FIFA territories and split fixtures from completed matches."""
    df = df[
        ~df['home_team'].isin(EXCLUDE_TEAMS) &
        ~df['away_team'].isin(EXCLUDE_TEAMS)
    ].copy()

    # Split completed matches from upcoming WC fixtures
    df_completed = df[df['home_score'].notna()].copy()
    df_fixtures  = df[df['home_score'].isna()].copy()

    df_completed['home_score'] = df_completed['home_score'].astype(int)
    df_completed['away_score'] = df_completed['away_score'].astype(int)

    return df_completed, df_fixtures


def split_data(df_completed):
    """Chronological train/test split — no shuffling to prevent leakage."""
    train = df_completed[df_completed['date'] < SPLIT_DATE].copy()
    test  = df_completed[df_completed['date'] >= SPLIT_DATE].copy()

    print(f"Train: {len(train):,} matches before {SPLIT_DATE}")
    print(f"Test:  {len(test):,} matches from {SPLIT_DATE} onwards")

    return train, test


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base_dir  = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "raw" / "results.csv"
    out_dir   = base_dir / "data" / "processed" / "modeling"

    print("Loading raw data...")
    df = load_raw_data(data_path)

    print("Removing non-FIFA territories and micronations...")
    df_completed, df_fixtures = clean_data(df)

    print("Splitting data to prevent information leakage...")
    train, test = split_data(df_completed)

    matrices = {
        "train":    train,
        "test":     test,
        "fixtures": df_fixtures,
    }

    package_matrices(matrices, out_dir)