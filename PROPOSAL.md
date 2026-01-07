# Project Proposal: What Makes a Netflix Hit? Exploring Ratings, Trends, and Show Features

## 1. Project Title and Category
**Title:** What Makes a Netflix Hit?  
**Category:** Data Analysis & Visualization

This project investigates which factors drive the success of Netflix TV shows in the United States. By combining Google Trends data, IMDb ratings, and show features, it aims to understand how audience interest, critical reception, and content characteristics influence a show’s likelihood of reaching the Netflix Top-10 chart.  
The project will include a Monte Carlo simulation to model uncertainty in these predictions, allowing for more realistic insight into how different factors interact.

---

## 2. Problem Statement / Motivation
Netflix releases hundreds of original series yearly, yet only a small number achieve major success. Shows like *Wednesday* or *Stranger Things* dominate the Top-10 charts, while others fade quickly.  
This project explores the question: **What differentiates a hit from a flop?**

By integrating multiple data sources—Netflix charts, IMDb ratings, and public search interest—the project will identify measurable predictors of success. Beyond static analysis, a Monte Carlo simulation will help visualize how random variation in these predictors (e.g., hype or reviews) could alter a show’s outcome, offering a richer understanding of uncertainty in media success.

---

## 3. Planned Approach and Technologies

### **Data Collection**
- **Base dataset:** *Netflix TV Shows* from Kaggle — all Netflix titles (2018–2024).  
- **Success labels:** *Netflix Top-10 (U.S.)* dataset — weeks in Top-10 and total hours viewed.  
- **Ratings:** IMDb dataset or Kaggle *Netflix Originals with IMDb Ratings*.  
- **Public interest:** Google Trends (`pytrends`) — average search index around release date.

### **Data Processing**
- Merge datasets in `pandas`, clean titles, and standardize variables.  
- Create dependent variables:
  - `in_top10` (1 = appeared, 0 = not appeared)  
- Independent variables:
  - `imdb_rating`, `avg_trend_score`, `genre`, `release_year`, `episode_count`.

### **Analysis & Modeling**
1. **Exploratory Analysis:**  
   Correlation plots, trend visualizations (`matplotlib`, `seaborn`).
2. **Regression Models:**  
   - Logistic regression → predict probability of appearing in Top-10.  
   - Linear regression → predict duration (`weeks_in_top10`).
3. **Monte Carlo Simulation:**  
   - Randomly vary predictors (e.g., ±10% IMDb rating, ±15% trend score).  
   - Run 1,000+ simulations to generate a distribution of predicted success.  
   - Visualize with histograms or fan charts.

### **Technologies**
Python (`pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `scikit-learn`, `pytrends`, `streamlit`).

---

## 4. Expected Challenges and How I’ll Address Them
- **Data completeness:** Some shows lack full trend or rating data → use imputation or remove outliers.  
- **Title inconsistencies:** Use fuzzy matching to align datasets.  
- **Imbalanced data:** Far fewer hits than misses → apply weighted models.  
- **Monte Carlo realism:** Use observed variance to define realistic randomness ranges.  

---

## 5. Success Criteria
The project will be successful if:
- A merged dataset of ≥50 Netflix shows is created, including both successful and unsuccessful ones.  
- Visualizations reveal clear relationships between predictors and popularity.  
- Regression models provide interpretable results.  
- The Monte Carlo simulation effectively demonstrates prediction uncertainty and sensitivity to input variables.  
- Results are reproducible and presented professionally (clean code, visualizations, documentation).

---
