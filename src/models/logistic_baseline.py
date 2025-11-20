
# simple logistic regression for Netflix Top-10 prediction.
# - loads the merged dataset
# - keeps only shows with Trends + year >= 2010
# - drops rows with missing features
# - time-based train/test split (<= 2022 vs >= 2023)
# - fits logistic regression
# - prints basic metrics and coefficients

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_project_root() -> Path:
    """
    Return the project root directory (where README.md lives).

    This file is in src/models/, so we go two levels up:
    models -> src -> project root
    """
    return Path(__file__).resolve().parents[2]


def load_data() -> pd.DataFrame:
    """Load the final merged CSV from the data folder."""
    root = get_project_root()
    data_path = root / "data" / "merged_shows_top10_US_imdb_trends.csv"
    df = pd.read_csv(data_path)
    return df


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for modelling:

    - keep only rows with non-missing avg_trend_score
    - keep only shows with release_year >= 2010
    - create log_imdb_num_votes
    - drop any remaining rows that have NaN in the features or target
    """
    df = df.copy()

    # 1) basic filters
    df = df[df["avg_trend_score"].notna()]
    df = df[df["release_year"] >= 2010]

    # 2) log-transform number of votes (add 1 to handle zeros)
    df["log_imdb_num_votes"] = np.log1p(df["imdb_num_votes"])

    # 3) drop rows with missing values in the columns we will use
    cols_needed = ["imdb_rating", "log_imdb_num_votes", "avg_trend_score", "in_top10"]
    before = len(df)
    df = df.dropna(subset=cols_needed)
    after = len(df)
    print(f"Dropped {before - after} rows with NaN in features/target.")
    print(f"Rows remaining for modelling: {after}")

    return df


def temporal_split(df: pd.DataFrame):
    """
    Time-based split:
    - train on shows released up to 2022
    - test on shows released in 2023 and later
    """
    train_df = df[df["release_year"] <= 2022]
    test_df = df[df["release_year"] >= 2023]
    return train_df, test_df


def main():
    # 1. Load and prepare data
    df = load_data()
    df_model = prepare_model_data(df)

    # sanity check
    print("Total rows after filtering & dropna:", len(df_model))

    # 2. Temporal train/test split
    train_df, test_df = temporal_split(df_model)
    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))

    # 3. Features and target
    feature_cols = ["imdb_rating", "log_imdb_num_votes", "avg_trend_score"]
    target_col = "in_top10"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # 4. Build and fit logistic regression
    clf = LogisticRegression(
        class_weight="balanced",  # handle class imbalance a bit
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # 5. Predictions on test data
    y_pred = clf.predict(X_test)

    # 6. Basic metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== Simple logistic regression on 2023+ test data ===")
    print("Accuracy :", round(acc, 3))
    print("Precision:", round(prec, 3))
    print("Recall   :", round(rec, 3))
    print("F1 score :", round(f1, 3))

    # 7. Coefficients: how each feature relates to hit probability
    print("\n=== Coefficients (one per feature) ===")
    for name, coef in zip(feature_cols, clf.coef_[0]):
        print(f"{name:20s} {coef: .4f}")


if __name__ == "__main__":
    main()
