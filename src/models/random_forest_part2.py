from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
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


# 1) Paths + small text processing

def get_project_root() -> Path:
    # .../src/models/random_forest_part2.py -> parents[2] is project root
    return Path(__file__).resolve().parents[2]


def _genre_tokenizer(text: str) -> list[str]:
    # Because genres are multi-label; I split "Drama, Crime, Action & Adventure"
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p]


# 2) Load + filter exactly like Part 2 setup (no Trends required)

def load_data() -> pd.DataFrame:
    root = get_project_root()
    data_path = root / "data" / "merged_shows_top10_US_imdb.csv"
    print(f"Loading dataset from: {data_path}")

    df = pd.read_csv(data_path)

    required = ["release_year", "in_top10", "genres", "language", "imdb_rating", "imdb_num_votes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    # basic types
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df["in_top10"] = pd.to_numeric(df["in_top10"], errors="coerce")
    df["in_top10"] = (df["in_top10"] > 0).astype(int)

    df["imdb_rating"] = pd.to_numeric(df["imdb_rating"], errors="coerce")
    df["imdb_num_votes"] = pd.to_numeric(df["imdb_num_votes"], errors="coerce")
    df["log_imdb_num_votes"] = np.log1p(df["imdb_num_votes"])

    # To keep modern era comparable with Part 1 and reduce missingness effects
    df = df[df["release_year"].notna() & (df["release_year"] >= 2010)].copy()

    # So that Part 2 keeps IMDb variables (engagement proxies) as a baseline reference
    df = df.dropna(subset=["imdb_rating", "log_imdb_num_votes"]).copy()

    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # To avoid leakage from future shows into training
    train_df = df[df["release_year"] <= 2022].copy()
    test_df = df[df["release_year"] >= 2023].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Temporal split failed (empty train or test). Check release_year values.")
    return train_df, test_df


def cap_top_languages(
    train_df: pd.DataFrame, test_df: pd.DataFrame, top_n: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    # To reduce very high-cardinality languages and keep model stable
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



# 3) Model: preprocessing + Random Forest

def build_model(random_state: int = 42) -> Pipeline:
    # Genres -> multi-hot (top K genre tokens learned from TRAIN only)
    genre_pipe = Pipeline(
        steps=[
            ("select", FunctionTransformer(lambda x: pd.Series(x.squeeze()).fillna(""), validate=False)),
            ("vec", CountVectorizer(
                tokenizer=_genre_tokenizer,
                preprocessor=None,
                token_pattern=None,
                binary=True,
                max_features=15,  # To keep only the most common genres
            )),
        ]
    )

    # Language (already capped to top 5 + other) -> one-hot
    lang_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Scale helps when combined with sparse one-hot in pipeline
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

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",  # Because hits are rare in Part 2
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(steps=[("pre", pre), ("rf", rf)])


def get_feature_names(model: Pipeline) -> np.ndarray:
    # ColumnTransformer.get_feature_names_out can fail depending on transformers,
    # so I build names manually (stable + simple).
    pre = model.named_steps["pre"]

    vec = pre.named_transformers_["genres"].named_steps["vec"]
    genre_names = np.array([f"genres__{g}" for g in vec.get_feature_names_out()])

    ohe = pre.named_transformers_["lang"].named_steps["ohe"]
    lang_names = np.array([f"lang__{x}" for x in ohe.get_feature_names_out(["language_top"])])

    num_names = np.array(["num__imdb_rating", "num__log_imdb_num_votes", "num__release_year"])

    return np.concatenate([genre_names, lang_names, num_names])



# 4) Evaluation + saving outputs

def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, values_format="d")
    plt.title("Random Forest (Part 2) — Confusion Matrix")
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
    plt.title("Random Forest (Part 2) — ROC Curve")
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
    plt.title("Random Forest (Part 2) — Precision–Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_feature_importance(
    names: np.ndarray, importances: np.ndarray, out_csv: Path, out_png: Path, top_n: int = 25
) -> None:
    imp_df = pd.DataFrame({"feature": names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df.to_csv(out_csv, index=False)

    top = imp_df.head(top_n).copy()
    
    plt.figure(figsize=(7, leading_lines(top_n)))
    plt.barh(range(len(top))[::-1], top["importance"].values)
    plt.yticks(range(len(top))[::-1], top["feature"].values)
    plt.xlabel("Feature importance (Gini)")
    plt.title(f"Random Forest (Part 2) — Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def leading_lines(n: int) -> float:
    # To keep plot readable without hardcoding huge figures
    return max(4.0, min(10.0, 0.25 * n + 2.5))


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

    print(f"Train size: {len(X_train)} | hits: {int(y_train.sum())} | hit-rate: {y_train.mean():.4f}")
    print(f"Test size : {len(X_test)}  | hits: {int(y_test.sum())}  | hit-rate: {y_test.mean():.4f}")

    model = build_model(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics (accuracy is shown, but for imbalanced data PR-AUC is important)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "random_state": int(random_state),
    }

    print("\n=== Random Forest Part 2 (2023+ test set) ===")
    for k, v in metrics.items():
        if k == "random_state":
            continue
        print(f"{k:10s}: {v:.3f}")
    print(f"Random state used: {random_state}")

    # Used to save outputs
    root = get_project_root()
    out_dir = root / "results" / "models" / "random_forest_part2"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    save_confusion_matrix(y_test.values, y_pred, out_dir / "confusion_matrix.png")
    save_roc_curve(y_test.values, y_prob, out_dir / "roc_curve.png")
    save_pr_curve(y_test.values, y_prob, out_dir / "pr_curve.png")

    # Feature importance
    names = get_feature_names(model)
    importances = model.named_steps["rf"].feature_importances_
    save_feature_importance(
        names,
        importances,
        out_csv=out_dir / "feature_importance.csv",
        out_png=out_dir / "feature_importance_top25.png",
        top_n=25,
    )

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    run(random_state=42)
