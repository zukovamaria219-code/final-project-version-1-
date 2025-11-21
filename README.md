Main questions of the project: 
RQ1 – Predictive accuracy:how well can we predict whether a U.S. Netflix TV show will ever enter the U.S. Top-10 based on audience response (IMDb + Google Trends) and basic metadata?
RQ2 – Drivers of success:which factors are most strongly associated with Top-10 success: IMDb rating, Google search interest, number of votes, or show characteristics (release year, etc.)?
RQ3 – Value of public interest vs ratings:does Google search interest provide additional predictive power beyond IMDb rating alone?
RQ4 – Model comparison:how do simple models (logistic regression) compare to more flexible models (Random Forest, XGBoost) in terms of accuracy and interpretability?



### Exploratory Data Analysis
 Data
All data used in this project are stored in the data/ folder.Raw / external sources (not fully tracked in GitHub due to size limits):
1. netflix_tv_shows_detailed_up_to_2025.csv – Netflix catalogue with show metadata.
2. 2025-11-17_global_weekly.xlsx – Netflix weekly Top-10 charts.
3. title.basics.tsv, title.ratings.tsv – IMDb title metadata and ratings.

Intermediate merge scripts:
1. merge_data.py – merges Netflix TV catalogue with US Top-10 info.
2. merge_with_imdb.py – adds IMDb ratings and vote counts.
3. add_trends.py – adds Google Trends search interest.
4. check_final_counts.py – prints counts of shows / hits / non-hits for sanity checks.

Final modeling dataset (used by EDA and models):
1. merged_shows_top10_US_imdb_trends.csv – one row per US TV show (release year ≥ 2010), with:
2. in_top10 (0/1) – whether the show ever reached the US TV Top-10
3. imdb_rating, imdb_num_votes
4. avg_trend_score – US Google Trends average search interest in the release year
plus basic Netflix metadata (title, year, genre, etc.)

To reproduce basic EDA (summary statistics, correlations, and plots), run:

```bash
python -m src.eda_basic

2.Installation 
pip install -r requirements.txt

3. Run EDA which generates summary stats + plots.
python -m src.eda_basic
outputs saved to results/eda/

4. Baseline logistic regression
python -m src.models.logistic_baseline

5. Logistic regression comparison (scaled vs non-scaled)
python -m src.models.logistic_compare

6. RQ3: Does Google Trends add predictive power?
python -m src.models.rq3_test

Outputs are saved in results/figures/ and results/models/
7. I performed a Multicollinearity check (VIF) to evaluate relationships between predictors.”
python -m src.models.check_vif
Outputs are saved in results/models/logistic_baseline
