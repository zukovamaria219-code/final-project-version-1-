# src/models/check_vif.py

"""
Check multicollinearity (VIF) for the logistic regression features.

I focus on:
- imdb_rating
- log_imdb_num_votes
- avg_trend_score

This uses the same filtering as the logistic models:
  * avg_trend_score not null
  * release_year >= 2010
  * log_imdb_num_votes = log(1 + imdb_num_votes)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------- 1. Helpers: project root + loading ----------

def get_project_root() -> Path:
   
    return Path(__file__).resolve().parents[2]


def load_full_data() -> pd.DataFrame:
   
    root = get_project_root()
    data_path = root / "data" / "merged_shows_top10_US_imdb_trends.csv"
    df = pd.read_csv(data_path)
    return df


# ---------- 2. Apply same filters as modelling ----------

def prepare_modeling_dataset(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()

    df = df[df["avg_trend_score"].notna()]
    df = df[df["release_year"] >= 2010]

    df["log_imdb_num_votes"] = np.log1p(df["imdb_num_votes"])

    return df


# ---------- 3. Compute VIF ----------

def compute_vif(df: pd.DataFrame, feature_names):
    
    X = df[feature_names].dropna().reset_index(drop=True)

    vif_rows = []
    for i, col in enumerate(X.columns):
        vif_val = variance_inflation_factor(X.values, i)
        vif_rows.append({"feature": col, "vif": float(vif_val)})

    vif_df = pd.DataFrame(vif_rows)
    return X.shape[0], vif_df


def main():
    df = load_full_data()
    df_model = prepare_modeling_dataset(df)

    feature_names = ["imdb_rating", "log_imdb_num_votes", "avg_trend_score"]

    n_rows, vif_df = compute_vif(df_model, feature_names)

    print(f"Rows used for VIF (after filtering and dropna): {n_rows}\n")
    print("=== VIF for logistic regression features ===")
    print(vif_df.to_string(index=False))

    # Save to results for the report
    root = get_project_root()
    results_dir = root / "results" / "models" / "logistic_baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    vif_df.to_csv(results_dir / "vif_logistic_features.csv", index=False)
    print(f"\nSaved VIF table to: {results_dir / 'vif_logistic_features.csv'}")


if __name__ == "__main__":
    main()
