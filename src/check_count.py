# src/check_top10_counts.py

from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    # one level above src/
    return Path(__file__).resolve().parents[1]


def main():
    root = get_project_root()
    data_path = root / "data" / "merged_shows_top10_US_imdb_trends.csv"

    df = pd.read_csv(data_path)

    counts = df["in_top10"].value_counts()
    shares = df["in_top10"].value_counts(normalize=True)

    print("Counts:")
    print(counts)
    print("\nShares:")
    print(shares)


if __name__ == "__main__":
    main()
