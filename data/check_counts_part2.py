import pandas as pd
import numpy as np

df = pd.read_csv("merged_shows_top10_US_imdb.csv")  

# Convert types like in the model script
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
df["in_top10"] = pd.to_numeric(df["in_top10"], errors="coerce")
df["in_top10"] = (df["in_top10"] > 0).astype(int)

df["imdb_rating"] = pd.to_numeric(df["imdb_rating"], errors="coerce")
df["imdb_num_votes"] = pd.to_numeric(df["imdb_num_votes"], errors="coerce")
df["log_imdb_num_votes"] = np.log1p(df["imdb_num_votes"])

print(f"Dataset shape (file): {df.shape}")
print(f"Columns: {list(df.columns)}")

# === OVERALL ===
y_all = df["in_top10"]
print("\n=== OVERALL (all shows in file) ===")
print(f"Total shows: {len(df)}")
print(f"Top-10 hits (in_top10 = 1): {int(y_all.sum())}")
print(f"Non-hits (in_top10 = 0): {len(df) - int(y_all.sum())}")
print(f"Hit rate: {y_all.mean():.4f}")

# === PART 2 MODEL FILTER (same as logistic_part2_combined.py) ===
subset = df[df["release_year"].notna() & (df["release_year"] >= 2010)].copy()
subset = subset.dropna(subset=["imdb_rating", "log_imdb_num_votes"]).copy()

y = subset["in_top10"]

print("\n=== PART 2 MODELLING SUBSET (year>=2010 + IMDb non-missing) ===")
print(f"Total shows in Part 2 subset: {len(subset)}")
print(f"Top-10 hits in subset: {int(y.sum())}")
print(f"Non-hits in subset: {len(subset) - int(y.sum())}")
print(f"Hit rate in subset: {y.mean():.4f}")

# === TEMPORAL SPLIT (same as logistic_part2_combined.py) ===
train = subset[subset["release_year"] <= 2022].copy()
test = subset[subset["release_year"] >= 2023].copy()

print("\n=== TEMPORAL SPLIT (Part 2) ===")
print(f"Train (<=2022): {len(train)} | hits: {int(train['in_top10'].sum())}")
print(f"Test  (>=2023): {len(test)} | hits: {int(test['in_top10'].sum())}")

if len(train) == 0 or len(test) == 0:
    print("WARNING: Temporal split produced an empty train or test set. Check release_year distribution.")
