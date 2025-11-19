# src/eda_descriptive.py

from pathlib import Path
import pandas as pd
import numpy as np


def get_project_root() -> Path:
    """Return project root (folder that contains data/, src/, results/)."""
    return Path(__file__).resolve().parents[1]


def load_full_data() -> pd.DataFrame:
    root = get_project_root()
    data_path = root / "data" / "merged_shows_top10_US_imdb_trends.csv"
    df = pd.read_csv(data_path)
    return df


def main():
    root = get_project_root()
    results_dir = root / "results" / "eda"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_full_data()

    print("Final dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print()

    # ---------------------------------------------------------------------
    # 1. OVERALL: full Netflix universe (context)
    # ---------------------------------------------------------------------
    counts_full = df["in_top10"].value_counts()
    shares_full = df["in_top10"].value_counts(normalize=True)

    print("=== OVERALL (all shows in file) ===")
    print(f"Total shows: {len(df)}")
    print(f"Top-10 hits (in_top10 = 1): {counts_full.get(1, 0)}")
    print(f"Non-hits (in_top10 = 0): {counts_full.get(0, 0)}")
    print("\nProportions:")
    print(shares_full.rename(index={0: "non_hit", 1: "hit"}))
    print()

    # Save to CSV for the report
    overall_counts = pd.DataFrame(
        {
            "count": counts_full,
            "proportion": shares_full,
        }
    )
    overall_counts.to_csv(results_dir / "overall_hit_proportions_full_dataset.csv")

    # ---------------------------------------------------------------------
    # 2. MODELING SUBSET: with Trends & release_year >= 2010
    # ---------------------------------------------------------------------
    df_model = df[df["avg_trend_score"].notna()].copy()
    df_model = df_model[df_model["release_year"] >= 2010].copy()

    counts_model = df_model["in_top10"].value_counts()
    shares_model = df_model["in_top10"].value_counts(normalize=True)

    print("=== MODELING DATASET (with trend score & year >= 2010) ===")
    print(f"Total shows: {len(df_model)}")
    print(f"Top-10 hits: {counts_model.get(1, 0)}")
    print(f"Non-hits: {counts_model.get(0, 0)}")
    print("\nProportions:")
    print(shares_model.rename(index={0: "non_hit", 1: "hit"}))
    print()

    model_counts = pd.DataFrame(
        {
            "count": counts_model,
            "proportion": shares_model,
        }
    )
    model_counts.to_csv(results_dir / "hit_proportions_model_dataset.csv")

    # ---------------------------------------------------------------------
    # 3. Numeric summary for key variables (modeling dataset)
    # ---------------------------------------------------------------------
    numeric_cols = [
        "imdb_rating",
        "imdb_num_votes",
        "avg_trend_score",
        "total_weeks_top10",
        "best_weekly_rank",
        "release_year",
    ]

    numeric_summary = df_model[numeric_cols].describe()
    numeric_summary.to_csv(results_dir / "numeric_summary_model_dataset.csv")

    print("=== Numeric summary (modeling dataset) ===")
    print(numeric_summary)
    print()

    # ---------------------------------------------------------------------
    # 4. Group stats: hits vs non-hits
    # ---------------------------------------------------------------------
    group_stats = (
        df_model
        .groupby("in_top10")[numeric_cols]
        .agg(["mean", "median", "std"])
        .rename(index={0: "non_hit", 1: "hit"})
    )
    group_stats.to_csv(results_dir / "group_stats_hit_vs_non_hit.csv")

    print("=== Group stats by Top-10 status (modeling dataset) ===")
    print(group_stats)
    print()

    # ---------------------------------------------------------------------
    # 5. Correlation matrix (includes in_top10 -> point-biserial correlations)
    # ---------------------------------------------------------------------
    corr_cols = numeric_cols + ["in_top10"]
    corr = df_model[corr_cols].corr()
    corr.to_csv(results_dir / "correlation_model_dataset.csv")

    print("=== Correlation matrix (modeling dataset) ===")
    print(corr["in_top10"].sort_values(ascending=False))
    print()

    # ---------------------------------------------------------------------
    # 6. Conditional hit probabilities: P(hit | bins)
    # ---------------------------------------------------------------------
    # Rating bins
    rating_bins = pd.cut(
        df_model["imdb_rating"],
        bins=[0, 5.5, 6.5, 7.5, 8.5, 10],
        include_lowest=True,
        right=False,
    )
    p_hit_by_rating = df_model.groupby(rating_bins)["in_top10"].mean()
    p_hit_by_rating.to_csv(results_dir / "p_hit_by_rating_bin.csv")

    # Trends bins
    trend_bins = pd.cut(
        df_model["avg_trend_score"],
        bins=[0, 10, 20, 30, 40, 60, 100],
        include_lowest=True,
        right=False,
    )
    p_hit_by_trend = df_model.groupby(trend_bins)["in_top10"].mean()
    p_hit_by_trend.to_csv(results_dir / "p_hit_by_trend_bin.csv")

    # IMDb votes bins (quartiles)
    votes_bins = pd.qcut(df_model["imdb_num_votes"], q=4)
    p_hit_by_votes = df_model.groupby(votes_bins)["in_top10"].mean()
    p_hit_by_votes.to_csv(results_dir / "p_hit_by_votes_bin.csv")

    print("=== P(hit | rating bin) ===")
    print(p_hit_by_rating)
    print("\n=== P(hit | trend bin) ===")
    print(p_hit_by_trend)
    print("\n=== P(hit | votes quartile) ===")
    print(p_hit_by_votes)


if __name__ == "__main__":
    main()
