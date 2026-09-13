"""
src/analysis.py
---------------
Pure-Python analysis functions (no plotting).
Each function receives a cleaned DataFrame and returns a summary DataFrame
or scalar value that the notebooks / visualisation layer can use.

Tasks covered
-------------
  Task 1  – City-wise restaurant analysis
  Task 2  – Online delivery analysis
  Task 3  – Price range analysis
"""

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 1 — City-wise Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def city_restaurant_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with restaurant count per city, sorted descending.

    Columns returned: City, Restaurant Count
    """
    counts = (
        df.groupby("City")["Restaurant ID"]
        .count()
        .reset_index()
        .rename(columns={"Restaurant ID": "Restaurant Count"})
        .sort_values("Restaurant Count", ascending=False)
        .reset_index(drop=True)
    )
    return counts


def city_avg_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return average Aggregate rating per city, sorted descending.

    Columns returned: City, Avg Rating, Restaurant Count
    """
    stats = (
        df.groupby("City")
        .agg(
            Avg_Rating=("Aggregate rating", "mean"),
            Restaurant_Count=("Restaurant ID", "count"),
        )
        .reset_index()
        .rename(columns={"Avg_Rating": "Avg Rating", "Restaurant_Count": "Restaurant Count"})
        .sort_values("Avg Rating", ascending=False)
        .reset_index(drop=True)
    )
    stats["Avg Rating"] = stats["Avg Rating"].round(2)
    return stats


def top_n_cities(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Return the top-N cities by restaurant count with their average rating.
    """
    counts  = city_restaurant_counts(df).head(n)
    ratings = city_avg_rating(df)[["City", "Avg Rating"]]
    result  = counts.merge(ratings, on="City", how="left")
    return result


def city_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full city summary: count, avg rating, avg votes, avg cost.
    """
    summary = (
        df.groupby("City")
        .agg(
            Restaurant_Count  =("Restaurant ID",           "count"),
            Avg_Rating        =("Aggregate rating",        "mean"),
            Avg_Votes         =("Votes",                   "mean"),
            Avg_Cost          =("Average Cost for two",    "mean"),
        )
        .reset_index()
        .rename(columns={
            "Restaurant_Count": "Restaurant Count",
            "Avg_Rating":       "Avg Rating",
            "Avg_Votes":        "Avg Votes",
            "Avg_Cost":         "Avg Cost",
        })
        .sort_values("Restaurant Count", ascending=False)
        .reset_index(drop=True)
    )
    for col in ["Avg Rating", "Avg Votes", "Avg Cost"]:
        summary[col] = summary[col].round(2)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 2 — Online Delivery Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def delivery_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count of restaurants with / without online delivery.

    Columns returned: Has Online delivery, Count, Percentage
    """
    counts = (
        df["Has Online delivery"]
        .value_counts()
        .reset_index()
        .rename(columns={"Has Online delivery": "Delivery Flag", "count": "Count"})
    )
    counts["Label"]      = counts["Delivery Flag"].map({1: "Has Online Delivery", 0: "No Online Delivery"})
    counts["Percentage"] = (counts["Count"] / counts["Count"].sum() * 100).round(2)
    return counts


def delivery_rating_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare average rating, votes, and cost between restaurants
    with and without online delivery.
    """
    comp = (
        df.groupby("Has Online delivery")
        .agg(
            Count       =("Restaurant ID",        "count"),
            Avg_Rating  =("Aggregate rating",     "mean"),
            Avg_Votes   =("Votes",                "mean"),
            Avg_Cost    =("Average Cost for two", "mean"),
        )
        .reset_index()
        .rename(columns={
            "Has Online delivery": "Has Online Delivery",
            "Count":      "Count",
            "Avg_Rating": "Avg Rating",
            "Avg_Votes":  "Avg Votes",
            "Avg_Cost":   "Avg Cost",
        })
    )
    comp["Has Online Delivery"] = comp["Has Online Delivery"].map({1: "Yes", 0: "No"})
    for col in ["Avg Rating", "Avg Votes", "Avg Cost"]:
        comp[col] = comp[col].round(2)
    return comp


def delivery_by_city(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    For each of the top-N cities (by restaurant count),
    return the percentage of restaurants offering online delivery.
    """
    top_cities = city_restaurant_counts(df).head(top_n)["City"].tolist()
    sub = df[df["City"].isin(top_cities)].copy()
    result = (
        sub.groupby("City")["Has Online delivery"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index()
        .rename(columns={"Has Online delivery": "Delivery %"})
        .sort_values("Delivery %", ascending=False)
        .reset_index(drop=True)
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 3 — Price Range Analysis
# ═══════════════════════════════════════════════════════════════════════════════

PRICE_LABELS = {1: "Budget (1)", 2: "Mid-Range (2)", 3: "Premium (3)", 4: "Luxury (4)"}


def price_range_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count restaurants per price range, with label and percentage.
    """
    counts = (
        df["Price range"]
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"Price range": "Price Range", "count": "Count"})
    )
    counts["Label"]      = counts["Price Range"].map(PRICE_LABELS)
    counts["Percentage"] = (counts["Count"] / counts["Count"].sum() * 100).round(2)
    return counts


def price_range_avg_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average rating and vote statistics per price range.
    """
    stats = (
        df.groupby("Price range")
        .agg(
            Count      =("Restaurant ID",        "count"),
            Avg_Rating =("Aggregate rating",     "mean"),
            Avg_Votes  =("Votes",                "mean"),
            Avg_Cost   =("Average Cost for two", "mean"),
        )
        .reset_index()
        .rename(columns={
            "Price range": "Price Range",
            "Count":       "Count",
            "Avg_Rating":  "Avg Rating",
            "Avg_Votes":   "Avg Votes",
            "Avg_Cost":    "Avg Cost",
        })
    )
    stats["Label"] = stats["Price Range"].map(PRICE_LABELS)
    for col in ["Avg Rating", "Avg Votes", "Avg Cost"]:
        stats[col] = stats[col].round(2)
    return stats


def best_price_range(df: pd.DataFrame) -> dict:
    """
    Return the price range with the highest average rating.
    """
    stats = price_range_avg_rating(df)
    best  = stats.loc[stats["Avg Rating"].idxmax()]
    return {
        "Price Range": int(best["Price Range"]),
        "Label":       best["Label"],
        "Avg Rating":  best["Avg Rating"],
        "Count":       int(best["Count"]),
    }


def price_delivery_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tabulate price range vs. online delivery availability.
    """
    ct = pd.crosstab(
        df["Price range"],
        df["Has Online delivery"].map({1: "Has Delivery", 0: "No Delivery"}),
        margins=True,
    )
    ct.index = [PRICE_LABELS.get(i, i) for i in ct.index]
    return ct


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERAL EDA helpers
# ═══════════════════════════════════════════════════════════════════════════════

def dataset_overview(df: pd.DataFrame) -> dict:
    """Return key dataset statistics as a dict."""
    return {
        "total_restaurants":  len(df),
        "unique_cities":       df["City"].nunique(),
        "unique_cuisines":     df["Cuisines"].nunique(),
        "avg_rating":          round(df["Aggregate rating"].mean(), 2),
        "avg_votes":           round(df["Votes"].mean(), 2),
        "avg_cost":            round(df["Average Cost for two"].mean(), 2),
        "pct_online_delivery": round(df["Has Online delivery"].mean() * 100, 2),
        "pct_table_booking":   round(df["Has Table booking"].mean() * 100, 2),
    }


def top_cuisines(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Top-N cuisines by restaurant count (using Primary Cuisine column)."""
    counts = (
        df["Primary Cuisine"]
        .value_counts()
        .head(n)
        .reset_index()
        .rename(columns={"Primary Cuisine": "Cuisine", "count": "Count"})
    )
    return counts


def rating_distribution(df: pd.DataFrame) -> pd.Series:
    """Distribution of Rating Text labels."""
    return df["Rating text"].value_counts()
