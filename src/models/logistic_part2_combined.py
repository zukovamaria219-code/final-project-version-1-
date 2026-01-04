from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


def get_project_root() -> Path:
    # .../src/models/logistic_part2_combined.py -> parents[2] is project root
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
    path = root / "data" / "merged_shows_top10_US_imdb.csv"
    print(f"Loading dataset from: {path}")

    df = pd.read_csv(path)

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

    # Modern era
    df = df[df["release_year"].notna() & (df["release_year"] >= 2010)].copy()

    # For the combined model, we require IMDb to exist
    df = df.dropna(subset=["imdb_rating", "log_imdb_num_votes"]).copy()

    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["release_year"] <= 2022].copy()
    test_df = df[df["release_year"] >= 2023].copy()
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Temporal split failed (empty train or test). Check your years in the dataset.")
    return train_df, test_df


def cap_top_languages(train_df: pd.DataFrame, test_df: pd.DataFrame, top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    # Pick top languages from TRAIN only to ensure no leakage
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


def build_model(random_state: int = 42) -> Pipeline:
    # Genres -> multi-hot, top 10 learned from TRAIN via vectorizer
    genre_pipe = Pipeline(
        steps=[
            ("select", FunctionTransformer(lambda x: pd.Series(x.squeeze()).fillna(""), validate=False)),
            ("vec", CountVectorizer(
                tokenizer=_genre_tokenizer,
                preprocessor=None,
                token_pattern=None,
                binary=True,
                max_features=10,  
            )),
        ]
    )

    # Language -> one-hot 
    lang_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Numeric features: imdb_rating, log_votes, release_year
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

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs",
    )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def get_feature_names_manual(model: Pipeline) -> np.ndarray:
    pre = model.named_steps["pre"]

    vec = pre.named_transformers_["genres"].named_steps["vec"]
    genre_names = np.array([f"genres__{g}" for g in vec.get_feature_names_out()])

    ohe = pre.named_transformers_["lang"].named_steps["ohe"]
    lang_names = ohe.get_feature_names_out(["language_top"])
    lang_names = np.array([f"lang__{x}" for x in lang_names])

    num_names = np.array(["num__imdb_rating", "num__log_imdb_num_votes", "num__release_year"])

    return np.concatenate([genre_names, lang_names, num_names])


def plot_top_coeffs(names: np.ndarray, coefs: np.ndarray, out_path: Path, top_n: int = 25) -> None:
    idx = np.argsort(np.abs(coefs))[::-1][:top_n]
    n = names[idx]
    v = coefs[idx]

    plt.figure()
    plt.barh(range(len(v))[::-1], v)
    plt.yticks(range(len(v))[::-1], n)
    plt.xlabel("Coefficient (log-odds)")
    plt.title(f"Top {top_n} coefficients (Logistic Part 2 Combined)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cm(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion matrix (Part 2 Combined)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(random_state: int = 42) -> None:
    df = load_data()
    train_df, test_df = temporal_split(df)

    train_df, test_df, top_langs = cap_top_languages(train_df, test_df, top_n=5)
    print(f"Top 5 languages (train): {top_langs}")

    features = ["genres", "language_top", "imdb_rating", "log_imdb_num_votes", "release_year"]
    X_train = train_df[features]
    y_train = train_df["in_top10"]
    X_test = test_df[features]
    y_test = test_df["in_top10"]

    print(f"Train size: {len(X_train)} | positives: {int(y_train.sum())}")
    print(f"Test size : {len(X_test)}  | positives: {int(y_test.sum())}")

    model = build_model(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # To diagnose how many positives code predicts
    print(f"Predicted positives: {int(y_pred.sum())} out of {len(y_pred)}")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) == 2 else float("nan")
    pr = average_precision_score(y_test, y_proba)

    print("\n=== Logistic regression Part 2 COMBINED (2023+ test set) ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1 score : {f1:.3f}")
    print(f"ROC-AUC  : {roc:.3f}")
    print(f"PR-AUC   : {pr:.3f}")

    # Coefficients
    names = get_feature_names_manual(model)
    coefs = model.named_steps["clf"].coef_[0]
    idx = np.argsort(np.abs(coefs))[::-1][:30]

    print("\n=== Top coefficients (by absolute value) ===")
    for n, c in zip(names[idx], coefs[idx]):
        print(f"{n:45s} {c: .4f}")

    # Save plots
    root = get_project_root()
    results_dir = root / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_top_coeffs(names, coefs, results_dir / "logistic_part2_combined_coefficients.png", top_n=25)
    plot_cm(y_test.values, y_pred, results_dir / "logistic_part2_combined_confusion_matrix.png")

    print(f"\nSaved plots to: {results_dir}")
    print(f"Random state used: {random_state}")


if __name__ == "__main__":
    main(random_state=42)
