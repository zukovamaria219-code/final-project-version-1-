import pandas as pd

# 1. Load the final dataset (make sure the extension is .csv, not .cvs)
df = pd.read_csv("merged_shows_top10_US_imdb_trends.csv")

print("Final dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Overall counts (all rows)
total = len(df)
hits = df["in_top10"].sum()
nonhits = total - hits

print("\n=== OVERALL (all shows in file) ===")
print(f"Total shows: {total}")
print(f"Top-10 hits (in_top10 = 1): {hits}")
print(f"Non-hits (in_top10 = 0): {nonhits}")

# 3. Counts for rows that HAVE a trend score
df_trend = df[df["avg_trend_score"].notna()]
total_trend = len(df_trend)
hits_trend = df_trend["in_top10"].sum()
nonhits_trend = total_trend - hits_trend

print("\n=== ONLY SHOWS WITH TREND SCORE (avg_trend_score not null) ===")
print(f"Total shows with trend data: {total_trend}")
print(f"Top-10 hits with trend data: {hits_trend}")
print(f"Non-hits with trend data: {nonhits_trend}")

# 4. (Optional) Restrict to release_year >= 2010
df_2010 = df_trend[df_trend["release_year"] >= 2010]
total_2010 = len(df_2010)
hits_2010 = df_2010["in_top10"].sum()
nonhits_2010 = total_2010 - hits_2010

print("\n=== WITH TREND DATA AND release_year >= 2010 ===")
print(f"Total shows: {total_2010}")
print(f"Top-10 hits: {hits_2010}")
print(f"Non-hits: {nonhits_2010}")
