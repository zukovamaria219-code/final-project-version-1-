# src/models/logistic_compare.py
#
# Compare two logistic regression models:
# 1) Simple (no scaling)
# 2) Pipeline with StandardScaler (more standard setup)
#
# Both use the same data, same temporal train/test split, and same features.

from pathlib import Path  
import numpy as np        
import pandas as pd       

from sklearn.linear_model import LogisticRegression               
from sklearn.preprocessing import StandardScaler                  
from sklearn.pipeline import Pipeline                             
from sklearn.metrics import (                                    
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ---------- 1. Helper: project root and data loading ----------

def get_project_root() -> Path:
    """
    Return the project root directory.
    This file is in src/models/, so we go two levels up: models -> src -> root
    """
    return Path(__file__).resolve().parents[2]  


def load_data() -> pd.DataFrame:
    """
    Load the final merged modelling dataset from data/.
    """
    root = get_project_root()                                      
    data_path = root / "data" / "merged_shows_top10_US_imdb_trends.csv"  
    df = pd.read_csv(data_path)                                    
    return df


# ---------- 2. Prepare modelling dataset ----------

def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and clean data for modelling:
    - keep rows with non-missing Trends
    - restrict to release_year >= 2010
    - create log_imdb_num_votes
    - drop rows with NaN in features/target
    """
    df = df.copy()                                                 

    # Basic filters
    df = df[df["avg_trend_score"].notna()]                         
    df = df[df["release_year"] >= 2010]                            

    # Log-transform vote counts (add 1 to handle zeros)
    df["log_imdb_num_votes"] = np.log1p(df["imdb_num_votes"])      

    # Drop rows with missing values in the columns we need
    cols_needed = ["imdb_rating", "log_imdb_num_votes", "avg_trend_score", "in_top10"]
    before = len(df)                                               
    df = df.dropna(subset=cols_needed)                             
    after = len(df)                                                
    print(f"Dropped {before - after} rows with NaN in features/target.")
    print(f"Rows remaining for modelling: {after}")

    return df


# ---------- 3. Temporal train/test split ----------

def temporal_split(df: pd.DataFrame, train_end_year: int = 2022):
    """
    Time-based split:
    - train on shows released up to train_end_year
    - test on shows released after train_end_year
    """
    train_df = df[df["release_year"] <= train_end_year]            
    test_df = df[df["release_year"] > train_end_year]              
    return train_df, test_df


# ---------- 4. Build models ----------

def build_simple_model() -> LogisticRegression:
    """
    Simple logistic regression (no scaling).
    """
    clf = LogisticRegression(
        class_weight="balanced",   
        max_iter=1000,             
        random_state=42,           
    )
    return clf


def build_scaled_pipeline() -> Pipeline:
    """
    Pipeline: StandardScaler + LogisticRegression.
    This is more standard because it rescales features before fitting.
    """
    scaler = StandardScaler()                                      
    clf = LogisticRegression(
        class_weight="balanced",   
        max_iter=1000,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("scaler", scaler),                                   
            ("clf", clf),  
       ]                                         
    )
    return pipe


# ---------- 5. Evaluate a model ----------

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str):
    """
    Fit the model, compute metrics on test set, and return a dict of results.
    """
    model.fit(X_train, y_train)                                   

    y_pred = model.predict(X_test)                               
    y_prob = model.predict_proba(X_test)[:, 1]                    

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)                          
    prec = precision_score(y_test, y_pred)                        
    rec = recall_score(y_test, y_pred)                            
    f1 = f1_score(y_test, y_pred)                                 
    roc_auc = roc_auc_score(y_test, y_prob)                       

    results = {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
    }

    return results


# ---------- 6. Main: run everything and compare ----------

def main():
    # 1) Load and prepare data
    df = load_data()                                              
    df_model = prepare_model_data(df)                             

    # 2) Train/test temporal split
    train_df, test_df = temporal_split(df_model, train_end_year=2022)
    print("Train rows:", len(train_df))
    print("Test rows :", len(test_df))

    # 3) Define features and target
    feature_cols = ["imdb_rating", "log_imdb_num_votes", "avg_trend_score"]  
    target_col = "in_top10"                                       

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # 4) Build both models
    simple_model = build_simple_model()                           
    scaled_model = build_scaled_pipeline()                        

    # 5) Evaluate both models
    simple_results = evaluate_model(simple_model, X_train, y_train, X_test, y_test, "simple_logistic")
    scaled_results = evaluate_model(scaled_model, X_train, y_train, X_test, y_test, "scaled_logistic")

    # 6) Collect results in a DataFrame
    results_df = pd.DataFrame([simple_results, scaled_results])

    # Reorder columns and round for a clean table
    cols_order = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    results_df = results_df[cols_order]
    results_df_rounded = results_df.round(3)

    # 7) Pretty print
    print("\n=== Logistic regression comparison (test set) ===")
    print(results_df_rounded.to_string(index=False))

    # 8) Save table to results/ for later use in report
    root = get_project_root()
    results_dir = root / "results" / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "logistic_compare_metrics.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved metrics table to: {out_path}")


if __name__ == "__main__":
    main() 

