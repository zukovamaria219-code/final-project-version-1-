from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from xgboost import XGBClassifier


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _genre_tokenizer(text: str) -> list[str]:
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p]


def load_data() -> pd.DataFrame:
    root = get_project_root()
    data_path = root / "data" / "merged_shows_top10_US_imdb.csv"
    print(f"Loading dataset from: {data_path}")

    df = pd.read_csv(data_path)

    required = ["release_year", "in_top10", "genres", "language", "imdb_rating", "imdb_num_votes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df["in_top10"] = pd.to_numeric(df["in_top10"], errors="coerce")
    df["in_top10"] = (df["in_top10"] > 0).astype(int)

    df["imdb_rating"] = pd.to_numeric(df["imdb_rating"], errors="coerce")
    df["imdb_num_votes"] = pd.to_numeric(df["imdb_num_votes"], errors="coerce")
    df["log_imdb_num_votes"] = np.log1p(df["imdb_num_votes"])

    # Part 2 filter (no Trends required)
    df = df[df["release_year"].notna() & (df["release_year"] >= 2010)].copy()
    df = df.dropna(subset=["imdb_rating", "log_imdb_num_votes"]).copy()

    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["release_year"] <= 2022].copy()
    test_df = df[df["release_year"] >= 2023].copy()
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Temporal split failed (empty train or test). Check release_year values.")
    return train_df, test_df


def cap_top_languages(
    train_df: pd.DataFrame, test_df: pd.DataFrame, top_n: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    top_langs = (
        train_df["language"].fillna("unknown")
        .value_counts()
        .head(top_n)
        .index
        .tolist()
    )

    def map_lang(s: pd.Series) -> pd.Series:
        s = s.fillna("unknown")
        return s.where(s.isin(top_langs), "other")

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["language_top"] = map_lang(train_df["language"])
    test_df["language_top"] = map_lang(test_df["language"])

    return train_df, test_df, top_langs


def build_model(random_state: int, scale_pos_weight: float) -> Pipeline:
    genre_pipe = Pipeline(
        steps=[
            ("select", FunctionTransformer(lambda x: pd.Series(x.squeeze()).fillna(""), validate=False)),
            ("vec", CountVectorizer(
                tokenizer=_genre_tokenizer,
                preprocessor=None,
                token_pattern=None,
                binary=True,
                max_features=15,
            )),
        ]
    )

    lang_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("genres", genre_pipe, ["genres"]),
            ("lang", lang_pipe, ["language_top"]),
            ("num", num_pipe, ["imdb_rating", "log_imdb_num_votes", "release_year"]),
        ]
    )

  
    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1,
        scale_pos_weight=scale_pos_weight,  
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
    )

    return Pipeline(steps=[("pre", pre), ("xgb", xgb)])


def get_feature_names(model: Pipeline) -> np.ndarray:
    pre = model.named_steps["pre"]

    vec = pre.named_transformers_["genres"].named_steps["vec"]
    genre_names = np.array([f"genres__{g}" for g in vec.get_feature_names_out()])

    ohe = pre.named_transformers_["lang"].named_steps["ohe"]
    lang_names = np.array([f"lang__{x}" for x in ohe.get_feature_names_out(["language_top"])])

    num_names = np.array(["num__imdb_rating", "num__log_imdb_num_votes", "num__release_year"])

    return np.concatenate([genre_names, lang_names, num_names])


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, values_format="d")
    plt.title("XGBoost (Part 2) — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("XGBoost (Part 2) — ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure()
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("XGBoost (Part 2) — Precision–Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_feature_importance(names: np.ndarray, importances: np.ndarray, out_dir: Path, top_n: int = 25) -> None:
    imp_df = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
    imp_df.to_csv(out_dir / "feature_importance.csv", index=False)

    top = imp_df.head(top_n)
    plt.figure(figsize=(7, max(4, 0.25 * top_n + 2.5)))
    plt.barh(range(len(top))[::-1], top["importance"].values)
    plt.yticks(range(len(top))[::-1], top["feature"].values)
    plt.xlabel("Feature importance")
    plt.title(f"XGBoost (Part 2) — Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance_top25.png", dpi=200)
    plt.close()


def run(random_state: int = 42) -> None:
    df = load_data()
    train_df, test_df = temporal_split(df)
    train_df, test_df, top_langs = cap_top_languages(train_df, test_df, top_n=5)
    print(f"Top 5 languages (train): {top_langs}")

    features = ["genres", "language_top", "imdb_rating", "log_imdb_num_votes", "release_year"]
    target = "in_top10"

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    print(f"Train size: {len(X_train)} | hits: {n_pos} | hit-rate: {y_train.mean():.4f}")
    print(f"Test size : {len(X_test)}  | hits: {int(y_test.sum())} | hit-rate: {y_test.mean():.4f}")
    print(f"scale_pos_weight (train): {scale_pos_weight:.2f}")

    model = build_model(random_state=random_state, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "random_state": int(random_state),
        "scale_pos_weight": float(scale_pos_weight),
    }

    print("\n=== XGBoost Part 2 (2023+ test set) ===")
    for k, v in metrics.items():
        if k in {"random_state"}:
            continue
        print(f"{k:15s}: {v:.3f}")
    print(f"Random state used: {random_state}")

    root = get_project_root()
    out_dir = root / "results" / "models" / "xgboost_part2"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    save_confusion_matrix(y_test.values, y_pred, out_dir / "confusion_matrix.png")
    save_roc_curve(y_test.values, y_prob, out_dir / "roc_curve.png")
    save_pr_curve(y_test.values, y_prob, out_dir / "pr_curve.png")

    names = get_feature_names(model)
    importances = model.named_steps["xgb"].feature_importances_
    save_feature_importance(names, importances, out_dir, top_n=25)

    print(f"\nSaved outputs to: {out_dir}")

if __name__ == "__main__":
    run(random_state=42)
