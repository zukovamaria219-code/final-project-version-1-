import time
import pandas as pd
from pytrends.request import TrendReq

# ---------- 1. Load the IMDb-enriched US TV dataset ----------
shows = pd.read_csv("merged_shows_top10_US_imdb.csv")
print("Loaded shows:", shows.shape)

if "in_top10" not in shows.columns:
    raise ValueError("Column 'in_top10' missing; did you use merged_shows_top10_US_imdb.csv?")

# Clean title
shows["title_clean"] = shows["title"].str.strip().str.lower()

# ---------- 2. To choose which shows to query for Trends ----------

recent = shows[shows["release_year"] >= 2010].copy()
hits_recent = recent[recent["in_top10"] == 1]
nonhits_recent = recent[recent["in_top10"] == 0].sample(
    n=len(hits_recent),
    random_state=42
) if len(recent[recent["in_top10"] == 0]) >= len(hits_recent) else recent[recent["in_top10"] == 0]

subset = pd.concat([hits_recent, nonhits_recent]).drop_duplicates()
print(f"Recent shows used for Trends queries: {len(subset)} "
      f"(hits: {len(hits_recent)}, non-hits: {len(nonhits_recent)})")

# ---------- 3. Set up pytrends ----------
pytrends = TrendReq(hl="en-US", tz=0)

def get_trend_score(title, year):
    """
    Average Google Trends score in the US during the year after release.
    Returns None if there is no data.
    """
    try:
        start_year = int(year)
    except (TypeError, ValueError):
        return None

    start = f"{start_year}-01-01"
    end = f"{start_year + 1}-01-01"
    timeframe = f"{start} {end}"

    try:
        pytrends.build_payload([title], timeframe=timeframe, geo="US")
        data = pytrends.interest_over_time()
        if data.empty or title not in data.columns:
            return None
        return float(data[title].mean())
    except Exception:
        return None

# ---------- 4. Query Trends for the subset ----------
scores = {}
print("Starting Google Trends queries...")

for idx, row in subset.iterrows():
    title = row["title"]
    year = row["release_year"]
    score = get_trend_score(title, year)
    scores[idx] = score
    
    time.sleep(1)

# ---------- 5. Attach scores back to the full shows dataframe ----------
shows["avg_trend_score"] = None  

for idx, score in scores.items():
    shows.loc[idx, "avg_trend_score"] = score

# ---------- 6. Print key counts ----------
total_shows = len(shows)
total_hits = shows["in_top10"].sum()

with_trend = shows[shows["avg_trend_score"].notna()]
n_with_trend = len(with_trend)
hits_with_trend = with_trend["in_top10"].sum()

print(f"\nTotal US TV shows (still): {total_shows}")
print(f"Total US TV shows in US Top-10 at least once: {total_hits}")
print(f"Shows with non-missing avg_trend_score: {n_with_trend}")
print(f"Top-10 hits among those with trend score: {hits_with_trend}")

# ---------- 7. Save as a NEW version ----------
shows.to_csv("merged_shows_top10_US_imdb_trends.csv", index=False)
print("\nSaved merged_shows_top10_US_imdb_trends.csv")
