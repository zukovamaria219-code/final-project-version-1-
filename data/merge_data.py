import pandas as pd

# ---------- 1. Load data ----------
shows = pd.read_csv("netflix_tv_shows_detailed_up_to_2025.csv")
global_weekly = pd.read_excel("2025-11-17_global_weekly.xlsx")

print("Shows shape:", shows.shape)
print("Global weekly shape:", global_weekly.shape)

print("\nShows columns:", shows.columns.tolist())
print("Global weekly columns:", global_weekly.columns.tolist())

# ---------- 2. Clean title keys (for matching later) ----------
shows["title_clean"] = shows["title"].str.strip().str.lower()
global_weekly["title_clean"] = global_weekly["show_title"].str.strip().str.lower()

# ---------- 2b. FILTER TO UNITED STATES IN BOTH FILES ----------

# (A) Weekly Top-10 data: keep only rows where the chart is for the US
# In this file, the country is called "United States"
global_us = global_weekly[global_weekly["country_name"] == "United States"]

print("\nUS weekly rows:", len(global_us))

# (B) TV-show metadata: keep only TV shows that involve the US
# The country string can be like "United States of America" or "Singapore, United States of America"
# str.contains("United States of America") keeps all rows where that substring appears anywhere
shows_us = shows[
    shows["country"].fillna("").str.contains("United States of America", case=False)
]

# Optional: only TV shows (not films)
shows_us = shows_us[shows_us["type"].str.contains("TV", case=False, na=False)]

print("US TV shows in metadata:", len(shows_us))

# ---------- 3. Keep only TV categories in WEEKLY data ----------
# Now we filter the US weekly chart to TV-only categories
global_us_tv = global_us[
    global_us["category"].str.contains("TV", case=False, na=False)
]

print("US weekly TV rows:", len(global_us_tv))
print("Unique US TV titles in global_us_tv:", global_us_tv["title_clean"].nunique())

# ---------- 4. Aggregate weekly data to ONE row per show ----------
top10_agg = (
    global_us_tv
    .groupby("title_clean", as_index=False)
    .agg(
        total_weeks_top10=("cumulative_weeks_in_top_10", "max"),  # total weeks in US Top-10
        best_weekly_rank=("weekly_rank", "min"),                  # best US rank (1 = best)
    )
)

# Binary label: 1 if it ever made the US Top-10, else 0
top10_agg["in_top10"] = (top10_agg["total_weeks_top10"] > 0).astype(int)

print("Rows in top10_agg (unique shows with US Top-10 info):", len(top10_agg))

# ---------- 5. Merge back to the US TV shows ----------
shows_merged = shows_us.merge(
    top10_agg,
    on="title_clean",
    how="left"      # keep all US TV shows, add US Top-10 info where available
)

# For shows never in Top-10, fill NaN with zeros
shows_merged["in_top10"] = shows_merged["in_top10"].fillna(0).astype(int)
shows_merged["total_weeks_top10"] = shows_merged["total_weeks_top10"].fillna(0).astype(int)

print("\nMerged shape:", shows_merged.shape)

# ---------- 6. Save merged data so you can inspect it ----------
SAVE_OUTPUT = False  # change to True when you do want to save

if SAVE_OUTPUT:
    shows_merged.to_csv("merged_shows_top10_US.csv", index=False)
    print("Saved merged_shows_top10_US.csv")
# --- 7. How many US TV shows were ever in the US Top-10? ---

num_in_top10 = shows_merged["in_top10"].sum()
num_total = len(shows_merged)

print(f"Total US TV shows in merged data: {num_total}")
print(f"US TV shows that were in US Top-10 at least once: {num_in_top10}")
