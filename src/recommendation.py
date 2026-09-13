"""
src/recommendation.py
---------------------
Task 5: Content-based Restaurant Recommendation System
using TF-IDF + Cosine Similarity.

How it works
------------
1. Build a "content string" for each restaurant combining:
   - Cuisines
   - City
   - Price range label
   - Rating category
   - Has Online delivery / Has Table booking
2. Vectorise with TF-IDF
3. Given user preferences (cuisine, city, price range, min rating),
   filter candidates and rank by cosine similarity to a synthetic
   query vector
4. Return top-N recommendations

Why content-based?
------------------
We have no user–item interaction history (no purchase logs),
so collaborative filtering is not applicable. Content-based
filtering on rich restaurant attributes is the natural choice.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise         import cosine_similarity
from src.analysis import PRICE_LABELS


# ═══════════════════════════════════════════════════════════════════════════════
class RestaurantRecommender:
    """
    Content-based restaurant recommendation system.

    Usage
    -----
    rec = RestaurantRecommender()
    rec.fit(df)                        # train on cleaned DataFrame
    results = rec.recommend(
        cuisine="Italian",
        city="New Delhi",
        price_range=2,
        min_rating=3.5,
        top_n=5
    )
    print(results)
    """

    # Columns shown in recommendations
    DISPLAY_COLS = [
        "Restaurant Name", "City", "Cuisines", "Primary Cuisine",
        "Price range", "Price Label", "Aggregate rating",
        "Rating Category", "Votes", "Has Online delivery",
        "Has Table booking", "Average Cost for two",
    ]

    def __init__(self):
        self.vectorizer   = None
        self.tfidf_matrix = None
        self.df           = None

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "RestaurantRecommender":
        """
        Build TF-IDF matrix from the cleaned DataFrame.
        """
        self.df = df.copy().reset_index(drop=True)
        self.df["Price Label"] = self.df["Price range"].map(PRICE_LABELS)

        # Build content strings
        self.df["_content"] = self.df.apply(self._build_content, axis=1)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["_content"])
        print(f"[recommender] Fitted on {len(self.df)} restaurants. "
              f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return self

    # ── Recommend ─────────────────────────────────────────────────────────────
    def recommend(
        self,
        cuisine:    str   = "",
        city:       str   = "",
        price_range: int  = 0,      # 0 = any; 1-4 = specific
        min_rating:  float = 0.0,
        top_n:       int  = 10,
        require_delivery: bool = False,
        require_booking:  bool = False,
    ) -> pd.DataFrame:
        """
        Return top-N restaurant recommendations as a DataFrame.

        Parameters
        ----------
        cuisine          : preferred cuisine (partial match OK)
        city             : preferred city (exact match preferred)
        price_range      : 1-4, or 0 for no preference
        min_rating       : minimum acceptable rating (0 to skip)
        top_n            : number of recommendations to return
        require_delivery : only return restaurants with online delivery
        require_booking  : only return restaurants with table booking
        """
        if self.df is None:
            raise RuntimeError("Call fit() before recommend()")

        # ── 1. Build query vector ─────────────────────────────────────────────
        price_label = PRICE_LABELS.get(price_range, "")
        query_parts = []
        if cuisine:     query_parts.append(cuisine)
        if city:        query_parts.append(city)
        if price_label: query_parts.append(price_label)
        query_str = " ".join(query_parts) if query_parts else "restaurant food"

        query_vec = self.vectorizer.transform([query_str])

        # ── 2. Cosine similarity across ALL restaurants ───────────────────────
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # ── 3. Hard filters ───────────────────────────────────────────────────
        mask = pd.Series([True] * len(self.df))

        if city.strip():
            city_mask = self.df["City"].str.lower().str.contains(
                city.strip().lower(), na=False)
            mask = mask & city_mask

        if cuisine.strip():
            cuisine_mask = self.df["Cuisines"].str.lower().str.contains(
                cuisine.strip().lower(), na=False)
            mask = mask & cuisine_mask

        if price_range > 0:
            mask = mask & (self.df["Price range"] == price_range)

        if min_rating > 0:
            mask = mask & (self.df["Aggregate rating"] >= min_rating)

        if require_delivery:
            mask = mask & (self.df["Has Online delivery"] == 1)

        if require_booking:
            mask = mask & (self.df["Has Table booking"] == 1)

        # ── 4. Rank filtered candidates ───────────────────────────────────────
        candidates = self.df[mask].copy()
        candidates["Similarity Score"] = scores[mask.values]

        # Rank by (similarity * 0.5 + normalised rating * 0.5)
        max_rating = self.df["Aggregate rating"].max()
        candidates["Composite Score"] = (
            0.5 * candidates["Similarity Score"]
            + 0.5 * (candidates["Aggregate rating"] / max_rating)
        )
        candidates = (
            candidates
            .sort_values("Composite Score", ascending=False)
            .drop_duplicates(subset=["Restaurant Name", "City"])
            .head(top_n)
            .reset_index(drop=True)
        )

        if candidates.empty:
            print("[recommender] No restaurants matched your filters. "
                  "Try relaxing the criteria.")
            return pd.DataFrame()

        # ── 5. Select display columns ─────────────────────────────────────────
        show_cols = [c for c in self.DISPLAY_COLS if c in candidates.columns]
        show_cols += ["Similarity Score", "Composite Score"]
        result = candidates[show_cols].copy()

        # Human-readable delivery/booking
        result["Online Delivery"] = result["Has Online delivery"].map({1: "Yes", 0: "No"})
        result["Table Booking"]   = result["Has Table booking"].map({1: "Yes", 0: "No"})
        result = result.drop(columns=["Has Online delivery", "Has Table booking"])

        result["Similarity Score"] = result["Similarity Score"].round(4)
        result["Composite Score"]  = result["Composite Score"].round(4)

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _build_content(row: pd.Series) -> str:
        """
        Combine restaurant attributes into a single string for TF-IDF.
        Repeat important fields to give them more weight.
        """
        price_label = PRICE_LABELS.get(row.get("Price range", 0), "")
        delivery    = "online delivery available" if row.get("Has Online delivery", 0) == 1 else ""
        booking     = "table booking available"  if row.get("Has Table booking",  0) == 1 else ""
        parts = [
            str(row.get("Cuisines", "")),          # full cuisines (all types listed)
            str(row.get("Primary Cuisine", "")),    # repeated for weight
            str(row.get("Primary Cuisine", "")),
            str(row.get("City", "")),
            str(row.get("City", "")),               # repeated for weight
            price_label,
            str(row.get("Rating Category", "")),
            delivery,
            booking,
        ]
        return " ".join(p for p in parts if p).lower()

    # ── Utility ───────────────────────────────────────────────────────────────
    def available_cities(self) -> list:
        if self.df is None:
            return []
        return sorted(self.df["City"].dropna().unique().tolist())

    def available_cuisines(self) -> list:
        if self.df is None:
            return []
        return sorted(self.df["Primary Cuisine"].dropna().unique().tolist())

    def top_rated_in_city(self, city: str, top_n: int = 5) -> pd.DataFrame:
        """Quick helper: top-rated restaurants in a given city."""
        sub = self.df[self.df["City"].str.lower() == city.lower()]
        return (
            sub.sort_values("Aggregate rating", ascending=False)
            .head(top_n)
            [["Restaurant Name", "Cuisines", "Aggregate rating",
              "Price range", "Votes"]]
            .reset_index(drop=True)
        )
