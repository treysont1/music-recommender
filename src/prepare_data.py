"""
Filters spotify_tracks.csv down to the top 1000 tracks by popularity.

Usage:
    python src/prepare_data.py
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
INPUT_FILE = ROOT / "data" / "spotify_tracks.csv"
OUTPUT_FILE = ROOT / "data" / "spotify_tracks.csv"
TOP_N = 1000


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    df = df.nlargest(TOP_N, "track_popularity")
    df.reset_index(drop=True, inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(df)} tracks to {OUTPUT_FILE}")
    print(f"Mood distribution:\n{df['mood'].value_counts().to_string()}")
    print(f"Genre distribution:\n{df['genre'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
