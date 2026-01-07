import pandas as pd

path = "title.basics.tsv"  

chunksize = 500_000  
title_type_counts = {}
tv_chunks = []       

print("Reading IMDb basics in chunks...")

for chunk in pd.read_csv(
    path,
    sep="\t",
    na_values="\\N",
    usecols=["tconst", "titleType", "primaryTitle"],  
    chunksize=chunksize
):
    # 1) update counts for titleType in this chunk
    vc = chunk["titleType"].value_counts()
    for tt, n in vc.items():
        title_type_counts[tt] = title_type_counts.get(tt, 0) + n

    # 2) keep tvSeries + tvMiniSeries rows
    tv_mask = chunk["titleType"].isin(["tvSeries", "tvMiniSeries"])
    tv_chunk = chunk[tv_mask]
    tv_chunks.append(tv_chunk)

print("Finished reading basics in chunks.")

# Combine all TV chunks into one smaller dataframe
imdb_tv = pd.concat(tv_chunks, ignore_index=True)

print("\nCounts by titleType (all types):")
for tt, n in sorted(title_type_counts.items(), key=lambda x: -x[1]):
    print(f"{tt:15s} {n}")

print("\nNumber of tvSeries + tvMiniSeries rows:", len(imdb_tv))

# Optional: save the TV-only subset to a smaller file for later use
imdb_tv.to_csv("imdb_tv_series_only.tsv", sep="\t", index=False)
print("Saved imdb_tv_series_only.tsv")
