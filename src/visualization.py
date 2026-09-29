"""
src/visualization.py
--------------------
Reusable plotting functions for every task.
All functions return the Matplotlib Figure object so callers
can either display it (plt.show()) or save it to disk.

Style convention
----------------
- Uses seaborn "whitegrid" theme throughout
- All charts include: title, axis labels, tight layout
- Colour palette: "viridis" for sequential, "Set2" for categorical
"""

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for notebooks too)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
import os
from pathlib import Path

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE_SEQ  = "viridis"
PALETTE_CAT  = "Set2"
PALETTE_DIV  = "RdYlGn"
# Anchor figures dir to project root so it works regardless of CWD
_ROOT   = Path(__file__).resolve().parent.parent
FIG_DIR = str(_ROOT / "notebooks" / "figures")


def _save(fig: plt.Figure, filename: str | None) -> None:
    """Save figure to FIG_DIR if filename is given."""
    if filename:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 1 — City-wise Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def plot_top_cities_bar(city_counts: pd.DataFrame,
                        top_n: int = 10,
                        save_as: str | None = None) -> plt.Figure:
    """Horizontal bar chart: Top-N cities by restaurant count."""
    data = city_counts.head(top_n).sort_values("Restaurant Count")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette(PALETTE_SEQ, len(data))
    ax.barh(data["City"], data["Restaurant Count"], color=colors)
    ax.set_xlabel("Number of Restaurants", fontsize=12)
    ax.set_ylabel("City", fontsize=12)
    ax.set_title(f"Top {top_n} Cities by Restaurant Count", fontsize=14, fontweight="bold")
    # Annotate bars
    for i, (val, city) in enumerate(zip(data["Restaurant Count"], data["City"])):
        ax.text(val + 5, i, str(val), va="center", fontsize=9)
    plt.tight_layout()
    _save(fig, save_as)
    return fig


def plot_city_avg_rating(city_ratings: pd.DataFrame,
                         top_n: int = 10,
                         save_as: str | None = None) -> plt.Figure:
    """Bar chart: Average rating for top-N cities."""
    data = city_ratings.head(top_n)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette(PALETTE_DIV, len(data))
    bars = ax.bar(data["City"], data["Avg Rating"], color=colors, edgecolor="white", width=0.6)
    ax.set_xlabel("City", fontsize=12)
    ax.set_ylabel("Average Rating", fontsize=12)
    ax.set_title(f"Top {top_n} Cities by Average Restaurant Rating", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.axhline(y=data["Avg Rating"].mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"Mean = {data['Avg Rating'].mean():.2f}")
    ax.legend(fontsize=10)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.04,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    _save(fig, save_as)
    return fig


def plot_city_dual_axis(top_cities_df: pd.DataFrame,
                        save_as: str | None = None) -> plt.Figure:
    """
    Dual-axis chart: restaurant count (bars) + avg rating (line)
    for the top cities.
    """
    data = top_cities_df.sort_values("Restaurant Count", ascending=False)
    fig, ax1 = plt.subplots(figsize=(13, 6))

    colors = sns.color_palette(PALETTE_SEQ, len(data))
    ax1.bar(data["City"], data["Restaurant Count"], color=colors,
            label="Restaurant Count", alpha=0.85)
    ax1.set_xlabel("City", fontsize=12)
    ax1.set_ylabel("Restaurant Count", fontsize=12, color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    plt.xticks(rotation=35, ha="right", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(data["City"], data["Avg Rating"], color="crimson",
             marker="o", linewidth=2, markersize=7, label="Avg Rating")
    ax2.set_ylabel("Average Rating", fontsize=12, color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.set_ylim(0, 5.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)
    ax1.set_title("Top Cities: Restaurant Count vs Average Rating", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_as)
    return fig


def plot_rating_distribution_hist(df: pd.DataFrame,
                                   save_as: str | None = None) -> plt.Figure:
    """Histogram of aggregate ratings across all restaurants."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["Aggregate rating"], bins=20, color="steelblue",
            edgecolor="white", alpha=0.85)
    ax.axvline(df["Aggregate rating"].mean(), color="red", linestyle="--",
               linewidth=2, label=f"Mean = {df['Aggregate rating'].mean():.2f}")
    ax.set_xlabel("Aggregate Rating", fontsize=12)
    ax.set_ylabel("Number of Restaurants", fontsize=12)
    ax.set_title("Distribution of Restaurant Ratings (Zomato Dataset)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, save_as)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 2 — Online Delivery Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def plot_delivery_pie(delivery_counts: pd.DataFrame,
                      save_as: str | None = None) -> plt.Figure:
    """Pie chart: Online delivery availability breakdown."""
    labels = delivery_counts["Label"]
    sizes  = delivery_counts["Count"]
    colors = ["#2ecc71", "#e74c3c"]
    explode = (0.05, 0)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, explode=explode,
        startangle=140, textprops={"fontsize": 12},
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
    ax.set_title("Online Delivery Availability\n(% of Total Restaurants)",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    _save(fig, save_as)
    return fig


def plot_delivery_rating_bar(comp_df: pd.DataFrame,
                              save_as: str | None = None) -> plt.Figure:
    """Grouped bar chart comparing avg rating, votes, cost by delivery status."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = [("Avg Rating", "Average Rating", "#3498db"),
               ("Avg Votes",  "Average Votes",  "#e67e22"),
               ("Avg Cost",   "Average Cost for Two", "#9b59b6")]

    for ax, (col, label, color) in zip(axes, metrics):
        bars = ax.bar(comp_df["Has Online Delivery"], comp_df[col],
                      color=color, width=0.5, edgecolor="white", alpha=0.9)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Has Online Delivery", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Online Delivery vs No Online Delivery — Key Metrics Comparison",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_as)
    return fig


def plot_delivery_by_city(city_delivery_df: pd.DataFrame,
                          save_as: str | None = None) -> plt.Figure:
    """Horizontal bar chart: % online delivery by city."""
    data = city_delivery_df.sort_values("Delivery %")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("RdYlGn", len(data))
    ax.barh(data["City"], data["Delivery %"], color=colors)
    ax.set_xlabel("Online Delivery (%)", fontsize=12)
    ax.set_ylabel("City", fontsize=12)
    ax.set_title("Online Delivery Penetration by City (Top 15 Cities)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 110)
    for i, val in enumerate(data["Delivery %"]):
        ax.text(val + 0.5, i, f"{val:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    _save(fig, save_as)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 3 — Price Range Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def plot_price_range_bar(price_dist: pd.DataFrame,
                         save_as: str | None = None) -> plt.Figure:
    """Bar chart: Restaurant count per price range."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette(PALETTE_CAT, len(price_dist))
    bars = ax.bar(price_dist["Label"], price_dist["Count"],
                  color=colors, edgecolor="white", width=0.6)
    ax.set_xlabel("Price Range", fontsize=12)
    ax.set_ylabel("Number of Restaurants", fontsize=12)
    ax.set_title("Restaurant Distribution by Price Range", fontsize=14, fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 10,
                str(h), ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_as)
    return fig


def plot_price_range_pie(price_dist: pd.DataFrame,
                         save_as: str | None = None) -> plt.Figure:
    """Pie chart: % of restaurants per price range."""
    colors = sns.color_palette(PALETTE_CAT, len(price_dist))
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        price_dist["Count"],
        labels=price_dist["Label"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax.set_title("Price Range Distribution\n(% of Total Restaurants)",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    _save(fig, save_as)
    return fig


def plot_price_avg_rating(price_stats: pd.DataFrame,
                          save_as: str | None = None) -> plt.Figure:
    """Bar chart: Average rating per price range."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette(PALETTE_DIV, len(price_stats))
    bars = ax.bar(price_stats["Label"], price_stats["Avg Rating"],
                  color=colors, edgecolor="white", width=0.55)
    ax.set_xlabel("Price Range", fontsize=12)
    ax.set_ylabel("Average Rating", fontsize=12)
    ax.set_title("Average Rating by Price Range", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 5.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.04,
                f"{h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_as)
    return fig


def plot_price_cost_rating(price_stats: pd.DataFrame,
                           save_as: str | None = None) -> plt.Figure:
    """Scatter: avg cost vs avg rating, sized by count, per price range."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette(PALETTE_CAT, len(price_stats))
    for i, row in price_stats.iterrows():
        ax.scatter(row["Avg Cost"], row["Avg Rating"],
                   s=row["Count"] / 5,
                   color=colors[i], alpha=0.85, edgecolors="black", linewidth=0.8,
                   label=row["Label"], zorder=5)
    ax.set_xlabel("Average Cost for Two", fontsize=12)
    ax.set_ylabel("Average Rating", fontsize=12)
    ax.set_title("Average Cost vs Average Rating\n(bubble size ∝ restaurant count)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Price Range", fontsize=10, title_fontsize=11)
    plt.tight_layout()
    _save(fig, save_as)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 4 — Rating Prediction
# ═══════════════════════════════════════════════════════════════════════════════

def plot_actual_vs_predicted(y_test, y_pred,
                              model_name: str = "Model",
                              save_as: str | None = None) -> plt.Figure:
    """Scatter plot: actual vs predicted ratings."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, color="steelblue",
               edgecolors="none", s=25, label="Predictions")
    min_val = min(float(min(y_test)), float(min(y_pred))) - 0.2
    max_val = max(float(max(y_test)), float(max(y_pred))) + 0.2
    ax.plot([min_val, max_val], [min_val, max_val],
            color="red", linewidth=2, linestyle="--", label="Perfect Prediction")
    ax.set_xlabel("Actual Rating", fontsize=12)
    ax.set_ylabel("Predicted Rating", fontsize=12)
    ax.set_title(f"Actual vs Predicted Ratings — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, save_as)
    return fig


def plot_feature_importance(importance_df: pd.DataFrame,
                             top_n: int = 15,
                             save_as: str | None = None) -> plt.Figure:
    """Horizontal bar chart: top-N feature importances."""
    data = importance_df.head(top_n).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette(PALETTE_SEQ, len(data))
    ax.barh(data["Feature"], data["Importance"], color=colors)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_as)
    return fig


def plot_model_comparison(results_df: pd.DataFrame,
                           metric: str = "R2 Score",
                           save_as: str | None = None) -> plt.Figure:
    """Bar chart comparing models on a given metric."""
    data = results_df.sort_values(metric, ascending=(metric != "R2 Score"))
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette(PALETTE_CAT, len(data))
    bars = ax.bar(data["Model"], data[metric], color=colors,
                  edgecolor="white", width=0.55)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"Model Comparison — {metric}", fontsize=14, fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                f"{h:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    _save(fig, save_as)
    return fig


def plot_residuals(y_test, y_pred,
                   model_name: str = "Model",
                   save_as: str | None = None) -> plt.Figure:
    """Residual plot: predicted vs residuals."""
    residuals = np.array(y_test) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.4, color="darkorange",
               edgecolors="none", s=25)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted Rating", fontsize=12)
    ax.set_ylabel("Residual (Actual − Predicted)", fontsize=12)
    ax.set_title(f"Residual Plot — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_as)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERAL EDA helpers
# ═══════════════════════════════════════════════════════════════════════════════

def plot_top_cuisines(cuisine_counts: pd.DataFrame,
                      save_as: str | None = None) -> plt.Figure:
    """Horizontal bar chart: top cuisines by restaurant count."""
    data = cuisine_counts.sort_values("Count")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette(PALETTE_CAT, len(data))
    ax.barh(data["Cuisine"], data["Count"], color=colors)
    ax.set_xlabel("Number of Restaurants", fontsize=12)
    ax.set_ylabel("Cuisine", fontsize=12)
    ax.set_title("Top Cuisines by Restaurant Count", fontsize=14, fontweight="bold")
    for i, val in enumerate(data["Count"]):
        ax.text(val + 2, i, str(val), va="center", fontsize=8)
    plt.tight_layout()
    _save(fig, save_as)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame,
                              save_as: str | None = None) -> plt.Figure:
    """Heatmap of numerical feature correlations."""
    num_cols = ["Aggregate rating", "Votes", "Average Cost for two",
                "Price range", "Has Online delivery", "Has Table booking"]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, ax=ax, annot_kws={"size": 10})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_as)
    return fig
