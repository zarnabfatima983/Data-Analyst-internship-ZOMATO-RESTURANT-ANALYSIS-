# 🍽️ Data Analyst Internship Project — Zomato Restaurant Analysis

A complete, professional-grade data analysis project built on a **real Zomato restaurant dataset**
(9,551 restaurants across 15 countries). The project covers five tasks that demonstrate the full
data analyst skill set: data cleaning, EDA, statistical analysis, machine learning, and
building a recommendation system.

---

## 📂 Project Structure

```
DATA_ANALYST_INTERNSHIP/
│
├── data/
│   ├── raw/
│   │   └── zomato.csv                  ← original Kaggle dataset
│   └── processed/
│       └── zomato_cleaned.csv          ← cleaned & engineered dataset
│
├── notebooks/
│   ├── 01_city_analysis.ipynb          ← Task 1
│   ├── 02_online_delivery_analysis.ipynb  ← Task 2
│   ├── 03_price_range_analysis.ipynb   ← Task 3
│   ├── 04_rating_prediction.ipynb      ← Task 4
│   └── 05_recommendation_system.ipynb  ← Task 5
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py                ← full cleaning pipeline
│   ├── analysis.py                     ← analysis & aggregation functions
│   ├── visualization.py                ← all chart functions (Matplotlib/Seaborn)
│   ├── rating_prediction.py            ← ML training & evaluation
│   └── recommendation.py              ← TF-IDF recommendation engine
│
├── app.py                              ← Streamlit dashboard (all 5 tasks)
├── download_data.py                    ← auto-download dataset from Kaggle
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Name** | Zomato Restaurants Dataset |
| **Source** | [Kaggle — shrutimehta/zomato-restaurants-data](https://www.kaggle.com/datasets/shrutimehta/zomato-restaurants-data) |
| **Raw records** | 9,551 restaurants |
| **After cleaning** | 7,403 restaurants |
| **Countries** | 15 |
| **Cities** | 141 |
| **Key columns** | Restaurant Name, City, Cuisines, Aggregate rating, Votes, Price range, Has Online delivery, Has Table booking, Average Cost for two |

---

## 🚀 Quick Start

### 1. Clone / download the project

```bash
git clone <your-github-url>
cd DATA_ANALYST_INTERNSHIP
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

```bash
python download_data.py
```

> The script uses `kagglehub` to automatically download `zomato.csv` into `data/raw/`.  
> If it fails, manually download from [Kaggle](https://www.kaggle.com/datasets/shrutimehta/zomato-restaurants-data)
> and place `zomato.csv` in `data/raw/`.

### 5. Run the data cleaning pipeline

```bash
python src/data_cleaning.py
```

This creates `data/processed/zomato_cleaned.csv`.

### 6. Open Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to `notebooks/` and open any notebook in order.

### 7. Launch the Streamlit App

```bash
streamlit run app.py
```

Opens a full interactive dashboard in your browser at `http://localhost:8501`.

---

## 📋 Task Summary

### Task 1 — City-wise Restaurant Analysis
**Notebook:** `notebooks/01_city_analysis.ipynb`

- New Delhi dominates with ~55% of all restaurants
- 141 unique cities analysed
- Top cities by count: New Delhi, Gurgaon, Noida, Faridabad
- Top cuisines: North Indian, Chinese, Fast Food
- Positive correlation between votes and rating (~0.30)
- **Charts:** Bar (top cities), Rating comparison, Dual-axis, Histogram, Correlation heatmap

---

### Task 2 — Online Delivery Analysis
**Notebook:** `notebooks/02_online_delivery_analysis.ipynb`

- Only **31.8%** of restaurants offer online delivery
- Restaurants without delivery have marginally higher avg rating (3.47 vs 3.38)
- Delivery restaurants get more votes (higher visibility)
- Budget restaurants lead delivery adoption; Luxury rarely delivers
- **Charts:** Pie, Grouped bar, Box plot, Violin plot, City delivery bar

---

### Task 3 — Price Range Analysis
**Notebook:** `notebooks/03_price_range_analysis.ipynb`

- Budget + Mid-Range = 74% of all restaurants
- Clear pattern: higher price → higher average rating (3.24 → 3.89)
- Premium (Price 3) restaurants have the most votes (~455 avg)
- Best value tier: **Premium (3)** — highest engagement + strong rating
- **Charts:** Bar, Pie, Rating bar, Bubble scatter, Box plot, Stacked bar

---

### Task 4 — Rating Prediction (Machine Learning)
**Notebook:** `notebooks/04_rating_prediction.ipynb`

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 0.3056 | 0.3990 | 0.4686 |
| Ridge Regression | 0.3039 | 0.3971 | 0.4736 |
| Random Forest | 0.2640 | 0.3597 | 0.5681 |
| **Gradient Boosting** | **0.2548** | **0.3470** | **0.5981** |

- Best model: **Gradient Boosting** (R² = 0.60)
- Top features: Votes, Price range, Average Cost, City, Primary Cuisine
- ~60% of rating variance explained — strong for subjective ratings
- **Charts:** Actual vs predicted, Residuals, Feature importances, Model comparison

---

### Task 5 — Restaurant Recommendation System
**Notebook:** `notebooks/05_recommendation_system.ipynb`  
**App:** `app.py` (Streamlit)

- **Technique:** Content-based filtering with TF-IDF + Cosine Similarity
- TF-IDF vocabulary: 3,302 terms from restaurant attributes
- Composite score = 0.5 × cosine similarity + 0.5 × normalised rating
- Filters: cuisine, city, price range, min rating, delivery, table booking
- Full interactive UI in Streamlit with expandable restaurant detail cards

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.11** | Core language |
| **Pandas** | Data loading, cleaning, aggregation |
| **NumPy** | Numerical operations |
| **Matplotlib** | Base plotting |
| **Seaborn** | Statistical visualisations |
| **Scikit-learn** | ML models, preprocessing, TF-IDF, cosine similarity |
| **SciPy** | Statistical significance tests (t-test) |
| **Streamlit** | Interactive web dashboard |
| **Jupyter Notebook** | Exploratory analysis notebooks |
| **kagglehub** | Dataset download |
| **Git / GitHub** | Version control |

---

## 📁 Data Cleaning Summary

| Step | Action | Rows affected |
|------|--------|---------------|
| Duplicates | Removed exact duplicate rows | 0 |
| Missing Cuisines | Filled with 'Unknown' | 9 |
| Unrated restaurants | Removed rows with rating = 0 | 2,148 |
| Cost outliers | Capped at Q3 + 3×IQR | 215 |
| Yes/No columns | Converted to 1/0 integer flags | — |
| Derived columns | Added Rating Category, Primary Cuisine | — |

---

## 💡 What I Learned

### From Each Task
| Task | Key Learning |
|------|-------------|
| Task 1 | City-level aggregation, groupby operations, multi-axis charts |
| Task 2 | Binary feature analysis, t-tests for significance, cross-tabulations |
| Task 3 | Distribution analysis, price-quality relationship, pie/bar visualisation |
| Task 4 | Full ML pipeline, feature engineering, model evaluation, hyperparameters |
| Task 5 | NLP (TF-IDF), cosine similarity, content-based filtering architecture |

### Python Libraries Used
- **Pandas & NumPy** — data manipulation, missing value handling, aggregations
- **Matplotlib & Seaborn** — 20+ professional chart types across 5 tasks
- **Scikit-learn** — `LinearRegression`, `Ridge`, `RandomForestRegressor`,
  `GradientBoostingRegressor`, `ColumnTransformer`, `Pipeline`, `TfidfVectorizer`,
  `cosine_similarity`
- **SciPy** — independent t-test for statistical significance
- **Streamlit** — multi-page interactive app with forms, sliders, expanders

### Data Analyst Skills Demonstrated
- Data wrangling and cleaning (handling NaNs, outliers, encoding)
- Exploratory Data Analysis (EDA)
- Statistical hypothesis testing
- Data visualisation and storytelling
- Machine learning (regression, preprocessing, evaluation)
- Natural language processing (TF-IDF vectorisation)
- Recommendation system design
- Building interactive dashboards

---

## 🎤 Internship Interview Talking Points

> **"Tell me about this project"**  
> "I built a complete data analysis pipeline on 9,551 real Zomato restaurant records. I cleaned the data, performed EDA across 5 dimensions, built a machine learning model that predicts restaurant ratings with 60% R², and created a TF-IDF content-based recommendation system — all wrapped in a Streamlit dashboard."

> **"What was your most important finding?"**  
> "The single strongest predictor of a restaurant's rating is the number of votes it receives — more popular restaurants are rated higher. Price range is second — premium restaurants consistently outrate budget ones. This suggests that a new restaurant should focus on getting early reviews and positioning itself in the premium tier."

> **"What ML model did you use and why?"**  
> "I compared four models. Gradient Boosting outperformed all others (R²=0.60, MAE=0.25) because it handles non-linear feature interactions well — for example, the combination of a specific city AND cuisine type matters more than either factor alone, which tree-based boosting captures naturally."

> **"How does your recommendation system work?"**  
> "It uses content-based filtering. I convert each restaurant's attributes into a TF-IDF document, then rank candidates by cosine similarity to the user's query. I blend that with a normalised rating score so the system doesn't just return obscure restaurants that happen to match the query — it recommends well-rated ones."

---

## 📜 License

Dataset: [CC0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) via Kaggle.  
Project code: MIT License.

---

*Built as part of a Data Analyst Internship — all analysis performed on real data, no mock statistics.*
# Data-Analyst-internship-ZOMATO-RESTURANT-ANALYSIS-
