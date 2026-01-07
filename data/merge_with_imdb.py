import pandas as pd

# ---------- 1. Load your US TV shows with Top-10 info ----------
shows = pd.read_csv("merged_shows_top10_US.csv")
print("US TV shows (original):", shows.shape)

# Safety: make sure the Top-10 flag exists
if "in_top10" not in shows.columns:
    raise ValueError("Column 'in_top10' not found in merged_shows_top10_US.csv")

# ---------- 2. Load IMDb TV-only subset + ratings ----------

imdb_tv = pd.read_csv("imdb_tv_series_only.tsv", sep="\t")

# Ratings file from IMDb datasets
ratings = pd.read_csv("title.ratings.tsv", sep="\t", na_values="\\N")

print("IMDb TV subset shape:", imdb_tv.shape)
print("IMDb ratings shape:", ratings.shape)

# Merge ratings into the TV subset
imdb_tv = imdb_tv.merge(ratings, on="tconst", how="left")

# ---------- 3. Clean titles for matching ----------
shows["title_clean"] = shows["title"].str.strip().str.lower()
imdb_tv["title_clean"] = imdb_tv["primaryTitle"].str.strip().str.lower()

# ---------- 4. Handle duplicate IMDb titles ----------

imdb_tv = (
    imdb_tv
    .sort_values("numVotes", ascending=False)
    .drop_duplicates(subset=["title_clean"])
)

print("IMDb TV after dedup on title_clean:", imdb_tv.shape)

# ---------- 5. Merge IMDb info into your US TV shows ----------
shows_imdb = shows.merge(
    imdb_tv[["title_clean", "averageRating", "numVotes"]],
    on="title_clean",
    how="left"
)

# Rename for clarity
shows_imdb = shows_imdb.rename(
    columns={
        "averageRating": "imdb_rating",
        "numVotes": "imdb_num_votes"
    }
)

print("\nAfter IMDb merge (all US TV shows):", shows_imdb.shape)

# ---------- 6. print key counts ----------
# 6a) Total US TV shows and number of Top-10 hits (this should stay >= 212)
total_shows = len(shows_imdb)
total_hits = shows_imdb["in_top10"].sum()

print(f"Total US TV shows after IMDb merge: {total_shows}")
print(f"US TV shows in US Top-10 at least once (in_top10 = 1): {total_hits}")

# 6b) How many have IMDb rating info
has_imdb = shows_imdb[shows_imdb["imdb_rating"].notna()]
n_with_imdb = len(has_imdb)
hits_with_imdb = has_imdb["in_top10"].sum()

print(f"\nShows with non-missing IMDb rating: {n_with_imdb}")
print(f"Top-10 hits among those with IMDb rating: {hits_with_imdb}")



# ---------- 7. Save  ----------
shows_imdb.to_csv("merged_shows_top10_US_imdb.csv", index=False)
print("\nSaved merged_shows_top10_US_imdb.csv")
