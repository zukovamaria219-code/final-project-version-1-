# Baseline XGBoost model for predicting Netflix Top-10 success.
# - loads final merged dataset
# - applies same filtering & feature engineering as logistic/RF
# - uses temporal split (<=2022 train, >=2023 test)
# - trains XGBoostClassifier
# - saves metrics, ROC curve, confusion matrix, feature importances

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from xgboost import XGBClassifier


# ---------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------

def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]  # models -> src -> root


def load_and_prepare_data():
    """Load merged data and apply same preprocessing as logistic/RF."""
    root = get_project_root()
    df = pd.read_csv(root / "data" / "merged_shows_top10_US_imdb_trends.csv")

    # Filters (same as logistic/RF)
    df = df[df["avg_trend_score"].notna()]
    df = df[df["release_year"] >= 2010]

    # Feature engineering
    df["log_imdb_num_votes"] = np.log1p(df["imdb_num_votes"])

    needed = ["imdb_rating", "log_imdb_num_votes", "avg_trend_score", "in_top10"]
    df = df.dropna(subset=needed)

    # Temporal split
    train = df[df["release_year"] <= 2022]
    test = df[df["release_year"] >= 2023]

    return train, test


# ---------------------------------------------------------
# 2. Save helpers (ROC, confusion matrix, feature importance)
# ---------------------------------------------------------

def save_roc_curve(y_test, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="XGBoost (AUC = {:.3f})".format(roc_auc_score(y_test, y_prob)))
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – XGBoost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_confusion_matrix(y_test, y_pred, save_path):
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["true_0_non_hit", "true_1_hit"],
        columns=["pred_0_non_hit", "pred_1_hit"]
    )
    cm_df.to_csv(save_path)


def save_feature_importance(model, feature_names, save_path_csv, save_path_plot):
    importance = model.feature_importances_

    # Save CSV
    pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    ).to_csv(save_path_csv, index=False)

    # Save plot
    plt.figure(figsize=(6, 4))
    order = np.argsort(importance)
    plt.barh(np.array(feature_names)[order], importance[order], color="skyblue")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path_plot, dpi=300)
    plt.close()


# ---------------------------------------------------------
# 3. Main training & evaluation
# ---------------------------------------------------------

def main():
    train_df, test_df = load_and_prepare_data()

    feature_cols = ["imdb_rating", "log_imdb_num_votes", "avg_trend_score"]
    target = "in_top10"

    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    # Create results folder
    root = get_project_root()
    results_dir = root / "results" / "models" / "xgboost_baseline"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Train XGBoost model
    # -----------------------
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    # -----------------------
    # Predict
    # -----------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # -----------------------
    # Metrics
    # -----------------------
    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    # Save metrics
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix
    save_confusion_matrix(
        y_test, y_pred,
        results_dir / "confusion_matrix.csv"
    )

    # Save ROC curve plot
    save_roc_curve(
        y_test, y_prob,
        results_dir / "roc_curve.png"
    )

    # Save feature importance
    save_feature_importance(
        model,
        feature_cols,
        results_dir / "feature_importance.csv",
        results_dir / "feature_importance.png"
    )

    # Console output
    print("\n=== XGBoost Baseline Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nResults saved to:")
    print(results_dir)


if __name__ == "__main__":
    main()
