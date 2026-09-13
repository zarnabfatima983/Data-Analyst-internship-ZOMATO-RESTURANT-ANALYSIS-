"""
src/data_cleaning.py
--------------------
Loads the raw Zomato CSV, performs comprehensive data cleaning,
and saves a processed version to data/processed/zomato_cleaned.csv.

Cleaning steps
--------------
1.  Load with correct encoding (latin-1)
2.  Standardise column names (strip spaces, title-case)
3.  Drop exact duplicate rows
4.  Handle missing values
     - Cuisines  → fill with 'Unknown'
     - Numeric   → no NaNs detected, but coerce + fill defensively
5.  Fix data types (Yes/No → 1/0 boolean integers)
6.  Remove restaurants with Aggregate rating = 0
    (0 means "not yet rated" in Zomato's system — not a real rating)
7.  Detect and cap cost outliers using IQR on 'Average Cost for two'
8.  Trim whitespace from all string columns
9.  Save cleaned dataset
"""

import os
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH       = os.path.join("data", "raw",       "zomato.csv")
PROCESSED_DIR  = os.path.join("data", "processed")
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "zomato_cleaned.csv")


# ─────────────────────────────────────────────────────────────────────────────
def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV with latin-1 encoding."""
    df = pd.read_csv(path, encoding="latin-1")
    print(f"[load]  Raw shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing spaces from column names."""
    df.columns = [c.strip() for c in df.columns]
    return df


# ─────────────────────────────────────────────────────────────────────────────
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    print(f"[dupes] Dropped {dropped} duplicate row(s). Remaining: {len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill or drop missing values.
    - Cuisines  → 'Unknown'  (only column with NaNs in this dataset)
    - Any other string column → 'Unknown'
    - Numeric columns → median fill (defensive)
    """
    # Cuisines
    if "Cuisines" in df.columns:
        missing_count = df["Cuisines"].isna().sum()
        df["Cuisines"] = df["Cuisines"].fillna("Unknown")
        print(f"[missing] Filled {missing_count} missing Cuisines with 'Unknown'")

    # Defensive: fill remaining string NaNs
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in str_cols:
        n = df[col].isna().sum()
        if n > 0:
            df[col] = df[col].fillna("Unknown")
            print(f"[missing] Filled {n} NaN(s) in '{col}' with 'Unknown'")

    # Defensive: fill numeric NaNs with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        n = df[col].isna().sum()
        if n > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[missing] Filled {n} NaN(s) in '{col}' with median {median_val:.2f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Yes/No columns to integer flags (1/0).
    Coerce Aggregate rating and Votes to numeric.
    """
    yes_no_cols = ["Has Table booking", "Has Online delivery", "Is delivering now"]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # Ensure numeric types
    for col in ["Aggregate rating", "Votes", "Average Cost for two", "Price range"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ─────────────────────────────────────────────────────────────────────────────
def remove_unrated(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where Aggregate rating == 0.
    In Zomato's system, 0 means the restaurant has not been rated yet —
    keeping these would distort average-rating calculations.
    """
    before = len(df)
    df = df[df["Aggregate rating"] > 0].copy()
    print(f"[rating] Removed {before - len(df)} unrated rows (rating = 0). Remaining: {len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def cap_cost_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme values in 'Average Cost for two' using the IQR method.
    Values above Q3 + 3*IQR are capped (not removed) to preserve data volume.
    """
    col = "Average Cost for two"
    if col not in df.columns:
        return df
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 3 * IQR
    outliers = (df[col] > upper).sum()
    df[col] = df[col].clip(upper=upper)
    print(f"[outlier] Capped {outliers} extreme cost values at {upper:.0f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def trim_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all string (object) columns."""
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    return df


# ─────────────────────────────────────────────────────────────────────────────
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a few useful derived columns used across multiple tasks.
    - Rating Category : groups rating into Low / Medium / High / Excellent
    - Primary Cuisine : first cuisine listed (before the first comma)
    """
    # Rating category
    def rating_category(r):
        if r < 3.0:
            return "Low"
        elif r < 3.5:
            return "Average"
        elif r < 4.0:
            return "Good"
        elif r < 4.5:
            return "Very Good"
        else:
            return "Excellent"

    df["Rating Category"] = df["Aggregate rating"].apply(rating_category)

    # Primary cuisine (first one listed)
    df["Primary Cuisine"] = df["Cuisines"].str.split(",").str[0].str.strip()

    return df


# ─────────────────────────────────────────────────────────────────────────────
def clean(raw_path: str = RAW_PATH, save: bool = True) -> pd.DataFrame:
    """
    Master cleaning pipeline.
    Returns the cleaned DataFrame and optionally saves it to disk.
    """
    df = load_raw(raw_path)
    df = standardise_columns(df)
    df = drop_duplicates(df)
    df = handle_missing(df)
    df = fix_dtypes(df)
    df = remove_unrated(df)
    df = cap_cost_outliers(df)
    df = trim_strings(df)
    df = add_derived_columns(df)

    print(f"\n[clean] Final cleaned shape: {df.shape}")
    print(f"[clean] Columns: {list(df.columns)}")

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"[clean] Saved → {PROCESSED_PATH}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
def load_clean(path: str = PROCESSED_PATH) -> pd.DataFrame:
    """
    Load the already-cleaned CSV from data/processed/.
    If it doesn't exist yet, run the full cleaning pipeline first.
    """
    if not os.path.exists(path):
        print("[load_clean] Processed file not found — running cleaning pipeline …")
        return clean()
    df = pd.read_csv(path)
    print(f"[load_clean] Loaded {df.shape[0]} rows × {df.shape[1]} cols from {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = clean()
    print("\nSample rows:")
    print(df[["Restaurant Name", "City", "Cuisines",
              "Aggregate rating", "Price range",
              "Has Online delivery", "Has Table booking"]].head(5).to_string(index=False))
