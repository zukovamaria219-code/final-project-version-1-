from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)


# 1. To get project root + load data


def get_project_root() -> Path:
    
    return Path(__file__).resolve().parents[2]


def load_data() -> pd.DataFrame:
    root = get_project_root()
    data_path = root / "data" / "merged_shows_top10_US_imdb_trends.csv"
    df = pd.read_csv(data_path)

    # log-transform votes
    df["log_imdb_num_votes"] = np.log1p(df["imdb_num_votes"])

    # Filter: Trends available + year >= 2010
    df = df[df["avg_trend_score"].notna()]
    df = df[df["release_year"] >= 2010]

    required = ["imdb_rating", "log_imdb_num_votes", "avg_trend_score", "in_top10"]
    df = df.dropna(subset=required)

    return df



# 2. Temporal split (train ≤ 2022, test ≥ 2023)


def temporal_split(df: pd.DataFrame):
    train = df[df["release_year"] <= 2022]
    test = df[df["release_year"] >= 2023]
    return train, test



# 3. Train + evaluate Random Forest


def evaluate_rf():
    df = load_data()
    train_df, test_df = temporal_split(df)

    features = ["imdb_rating", "log_imdb_num_votes", "avg_trend_score"]
    target = "in_top10"

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42
    )

    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    
    # Save results
  
    root = get_project_root()
    out_dir = root / "results" / "models" / "random_forest_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title("Random Forest — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("Random Forest — ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png")
    plt.close()

    # Feature importance
    importances = rf.feature_importances_
    imp_df = pd.DataFrame({
        "feature": features,
        "importance": importances
    }).sort_values("importance", ascending=False)

    imp_df.to_csv(out_dir / "feature_importance.csv", index=False)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.barh(imp_df["feature"], imp_df["importance"], color="teal")
    plt.title("Random Forest — Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png")
    plt.close()

    print("\n=== Random Forest Baseline Results ===")
    print(metrics)
    print(f"\nSaved outputs to: {out_dir}")



# Entry point

if __name__ == "__main__":
    evaluate_rf()
