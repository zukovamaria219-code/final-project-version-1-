from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def get_project_root() -> Path:
    """Return the root directory of the project (where README.md lives)."""
    return Path(__file__).resolve().parents[1]


def load_data() -> pd.DataFrame:
    root = get_project_root()
    data_path = root / "data" / "merged_shows_top10_US_imdb_trends.csv"
    df = pd.read_csv(data_path)
    return df


def run_eda():
    root = get_project_root()
    results_dir = root / "results" / "eda"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()

    # --- 1. Basic info (dimensions, dtypes) ---
    with open(results_dir / "data_info.txt", "w") as f:
        f.write(f"Shape: {df.shape}\n\n")
        f.write(str(df.dtypes))

    # --- 2. Summary stats for numeric variables ---
    numeric_summary = df.describe()
    numeric_summary.to_csv(results_dir / "numeric_summary.csv", index=True)

    # --- 3. Correlation matrix for numeric variables ---
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    corr = df[numeric_cols].corr()
    corr.to_csv(results_dir / "correlation_numeric.csv", index=True)

    # --- 4. Histograms for a few key features ---
    # IMDb rating vs in_top10
    fig, ax = plt.subplots()
    df[df["in_top10"] == 1]["imdb_rating"].hist(ax=ax, alpha=0.7)
    df[df["in_top10"] == 0]["imdb_rating"].hist(ax=ax, alpha=0.7)
    ax.set_xlabel("IMDb rating")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of IMDb rating by Top-10 status")
    ax.legend(["Top-10 hit", "Non-hit"])
    fig.tight_layout()
    fig.savefig(results_dir / "hist_imdb_rating_by_top10.png")
    plt.close(fig)

    # Google trends vs in_top10
    fig, ax = plt.subplots()
    df[df["in_top10"] == 1]["avg_trend_score"].hist(ax=ax, alpha=0.7)
    df[df["in_top10"] == 0]["avg_trend_score"].hist(ax=ax, alpha=0.7)
    ax.set_xlabel("Average Google Trends score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Trend score by Top-10 status")
    ax.legend(["Top-10 hit", "Non-hit"])
    fig.tight_layout()
    fig.savefig(results_dir / "hist_trends_by_top10.png")
    plt.close(fig)

    # --- 5. Simple scatter: rating vs trends, colored by hit ---
    fig, ax = plt.subplots()
    hits = df[df["in_top10"] == 1]
    non_hits = df[df["in_top10"] == 0]
    ax.scatter(
        non_hits["imdb_rating"],
        non_hits["avg_trend_score"],
        alpha=0.5,
        label="Non-hit",
    )
    ax.scatter(
        hits["imdb_rating"],
        hits["avg_trend_score"],
        alpha=0.7,
        label="Top-10 hit",
    )
    ax.set_xlabel("IMDb rating")
    ax.set_ylabel("Average Google Trends score")
    ax.set_title("IMDb rating vs Trends, colored by Top-10")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "scatter_rating_vs_trends_by_top10.png")
    plt.close(fig)


if __name__ == "__main__":
    run_eda()
import pandas as pd

