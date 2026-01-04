# ============================================
# README (bash-style runbook)
# What Makes a Netflix Hit? — U.S. Top-10 TV Prediction
# ============================================

# ----------------------------
# 0. Project goal
# ----------------------------
# Predict whether a U.S. Netflix TV show will ever enter the U.S. weekly Top-10.
# Data sources: Netflix Top-10 histories + Netflix metadata + IMDb rating/votes + (optional) Google Trends.
# Evaluation: out-of-time split (train release_year <= 2022, test 2023–2024).
# Models: Logistic Regression, Random Forest, XGBoost. RQ3 compares IMDb-only vs IMDb+Trends.

# ----------------------------
# 1. Clone repository
# ----------------------------
git clone https://github.com/zukovamaria219-code/final-project-version-1-
cd final-project1

# ----------------------------
# 2. Create environment + install dependencies
# ----------------------------
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows (PowerShell/CMD)
pip install -r requirements.txt

# ----------------------------
# 3. Data (recommended path)
# ----------------------------
# The repo is designed to run out-of-the-box using final merged CSVs in ./data/:
#   Part 1: data/merged_shows_top10_US_imdb_trends.csv
#   Part 2: data/merged_shows_top10_US_imdb.csv
#
# Verify they exist:
ls -lh data/merged_shows_top10_US_imdb_trends.csv
ls -lh data/merged_shows_top10_US_imdb.csv

# ----------------------------
# 4. Quickstart (run everything)
# ----------------------------
# Runs EDA + main models and writes artifacts to ./results/
python main.py

# Outputs (after running):
#   results/eda/
#   results/figures/
#   results/models/

# ----------------------------
# 5. Run components individually (optional)
# ----------------------------

# --- EDA ---
python -m src.eda_basic
python -m src.eda_descriptive

# --- Part 1 models (IMDb + Trends subset) ---
python -m src.models.logistic_baseline
python -m src.models.logistic_compare
python -m src.models.random_forest_baseline
python -m src.models.xgboost_baseline
python -m src.models.rq3_test
python -m src.models.check_vif

# --- Part 2 models (IMDb + metadata; larger sample) ---
python -m src.models.logistic_part2_combined
python -m src.models.random_forest_part2
python -m src.models.xgboost_part2

# ----------------------------
# 6. (Optional) Rebuild datasets from raw sources
# ----------------------------
# Raw data links (fill in manually):
#   Netflix catalogue (Kaggle): <https://www.kaggle.com/datasets/bhargavchirumamilla/netflix-movies-and-tv-shows-till-2025>
#   Netflix Top-10 charts (Tudum): <https://www.netflix.com/tudum/top10>
#   IMDb datasets: <https://developer.imdb.com/non-commercial-datasets/>
#   Google Trends: generated via script (API limits may apply)
#
# Place raw files into ./data/ (or adjust script paths), then run:
python data/merge_data.py
python data/merge_with_imdb.py
python data/add_trends.py
python data/check_final_counts.py
python data/check_counts_part2.py

# ----------------------------
# 7. Reproducibility notes (what matters)
# ----------------------------
# - Temporal split for leakage control: train release_year <= 2022, test 2023–2024
# - Fixed random seed: random_state=42 (or equivalent)
# - Imbalance handling:
#     sklearn: class_weight="balanced"
#     XGBoost: scale_pos_weight (from training label ratio)

# ----------------------------
# 8. Tests (optional)
# ----------------------------
pytest -q

# ----------------------------
# 9. Troubleshooting (common)
# ----------------------------
# Missing file error:
#   - confirm you're in project root and the CSVs exist in ./data/
#   - run: ls data/
#
# Package error:
#   - activate venv and reinstall: pip install -r requirements.txt
#
# XGBoost install issues:
#   pip install -U pip setuptools wheel
#   pip install xgboost
