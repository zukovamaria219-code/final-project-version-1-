
# Simple logistic regression for Netflix Top-10 prediction.
# - loads the merged dataset
# - keeps only shows with Trends + year >= 2010
# - drops rows with missing features
# - time-based train/test split (<= 2022 vs >= 2023)
# - fits logistic regression
# - prints basic metrics and coefficients
# - saves coefficient plot and confusion matrix plot

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# --- 1. Setup & Data Loading (Same logic as your baseline) ---

def get_project_root() -> Path:
    """Return project root (one level above src/)."""
    return Path(__file__).resolve().parents[2]


def load_and_prep_data():
    """
    Load final merged CSV and prepare modelling sample:
    - avg_trend_score not null
    - release_year >= 2010
    - log_imdb_num_votes
    - temporal split: train <= 2022, test >= 2023
    """
    root = get_project_root()
    data_path = root / "data" / "merged_shows_top10_US_imdb_trends.csv"

    # Load
    df = pd.read_csv(data_path)

    # Filter & transform
    df = df[df["avg_trend_score"].notna()]
    df = df[df["release_year"] >= 2010]
    df["log_imdb_num_votes"] = np.log1p(df["imdb_num_votes"])

    # Drop NaNs in the columns we need
    cols_needed = ["imdb_rating", "log_imdb_num_votes", "avg_trend_score", "in_top10"]
    df = df.dropna(subset=cols_needed)

    # Temporal split
    train_df = df[df["release_year"] <= 2022]
    test_df = df[df["release_year"] >= 2023]

    return train_df, test_df


# --- 2. Plotting Functions ---

def plot_coefficients(model, feature_names, save_path):
    """
    Creates a horizontal bar chart of the model coefficients.
    """
    coefs = model.coef_[0]

    coef_df = (
        pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
        .sort_values(by="Coefficient", ascending=True)
    )

    # Color bars based on positive/negative direction
    colors = ["#ff4b4b" if x < 0 else "#4bff4b" for x in coef_df["Coefficient"]]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)

    plt.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    plt.title("Logistic Regression Coefficients\n(Drivers of Top-10 Success)", fontsize=14)
    plt.xlabel("Coefficient Value (Log-Odds)", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved coefficient plot to: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Creates a heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted: No", "Predicted: Hit"],
        yticklabels=["Actual: No", "Actual: Hit"],
    )

    plt.title("Confusion Matrix (Test Set 2023+)", fontsize=14)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved confusion matrix to: {save_path}")
    plt.close()


# --- 3. Main Execution ---

def main():
    # A. Prepare data
    train_df, test_df = load_and_prep_data()

    features = ["imdb_rating", "log_imdb_num_votes", "avg_trend_score"]
    target = "in_top10"

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # B. Train model
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # C. Predict
    y_pred = clf.predict(X_test)

    # D. Print metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("=== Simple logistic regression (2023+ test set) ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1 score : {f1:.3f}")

    # E. Print coefficients
    print("\n=== Coefficients (log-odds) ===")
    for name, coef in zip(features, clf.coef_[0]):
        print(f"{name:20s} {coef: .4f}")

    # F. Save plots
    root = get_project_root()
    results_dir = root / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_coefficients(clf, features, results_dir / "logistic_coefficients.png")
    plot_confusion_matrix(y_test, y_pred, results_dir / "logistic_confusion_matrix.png")


if __name__ == "__main__":
    main()
