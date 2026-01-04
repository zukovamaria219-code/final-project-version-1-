import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from pathlib import Path

# --- 1. Setup & Data Loading ---
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_and_prep_data():
    root = get_project_root()
    data_path = root / "data" / "merged_shows_top10_US_imdb_trends.csv"
    
    # Load data
    df = pd.read_csv(data_path)
    
    # 1. To create the Log Feature 
    # I add 1 to avoid log(0) errors, though log1p does this automatically
    df["log_imdb_num_votes"] = np.log1p(df["imdb_num_votes"])
    
    # 2. Define all columns I might need for any model
    # I must drop NaNs from all of these so both models use the exact same rows
    cols_needed = [
        "imdb_rating", 
        "log_imdb_num_votes", 
        "avg_trend_score", 
        "in_top10", 
        "release_year"
    ]
    
    # 3. To Keep only rows where all needed columns have data
    before_drop = len(df)
    df = df.dropna(subset=cols_needed)
    after_drop = len(df)
    
    print(f"Dropped {before_drop - after_drop} rows containing missing values (NaN).")
    print(f"Rows remaining: {after_drop}")

    # 4. To keep modern shows only
    df = df[df["release_year"] >= 2010]
    
    # 5. Temporal Split
    train = df[df["release_year"] <= 2022]
    test = df[df["release_year"] > 2022]
    
    return train, test

# --- 2. Model Builder Helper ---
def train_and_score(X_train, y_train, X_test, y_test):
    """
    Trains a scaled Logistic Regression and returns the ROC-AUC score.
    """
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # To get probabilities for the positive class (class 1)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate AUC
    auc = roc_auc_score(y_test, y_probs)
    return auc

# --- 3. The Experiment ---
def main():
    print("--- Loading Data for RQ3 Analysis ---")
    train_df, test_df = load_and_prep_data()
    target = "in_top10"
    
    # To define the two feature sets
    features_baseline = ["imdb_rating", "log_imdb_num_votes"]
    features_full     = ["imdb_rating", "log_imdb_num_votes", "avg_trend_score"]
    
    print(f"\nTest Set Size: {len(test_df)} shows (2023-2024)")
    
    # --- Model A: Baseline (IMDb Only) ---
    print("Training Model A (Baseline)...")
    auc_baseline = train_and_score(
        train_df[features_baseline], train_df[target],
        test_df[features_baseline], test_df[target]
    )
    
    # --- Model B: Full (IMDb + Trends) ---
    print("Training Model B (Full + Trends)...")
    auc_full = train_and_score(
        train_df[features_full], train_df[target],
        test_df[features_full], test_df[target]
    )
    
    # --- 4. Results & Conclusion ---
    lift = auc_full - auc_baseline
    
    print("\n=== RQ3 Analysis Results ===")
    print(f"Model A (IMDb Only) ROC-AUC:     {auc_baseline:.4f}")
    print(f"Model B (IMDb + Trends) ROC-AUC: {auc_full:.4f}")
    print("-" * 40)
    print(f"Performance Gain (Lift):         {lift:+.4f}")
    
    print("\n--- Conclusion for Report ---")
    if lift > 0.01:
        print("✅ Google Trends ADDS significant predictive power.")
        print("   Interpretation: Public search interest captures signals that ratings miss.")
    elif lift < -0.01:
        print("❌ Google Trends HURTS the model (noise).")
    else:
        print("⚠️ Google Trends adds MINIMAL/NO unique value.")
        print("   Interpretation: IMDb votes/ratings likely already capture the popularity signal.")

if __name__ == "__main__":
    main()