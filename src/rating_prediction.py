"""
src/rating_prediction.py
------------------------
Task 4: Build, train, evaluate and compare ML models that predict
the Aggregate rating of a restaurant.

Pipeline
--------
1. Feature selection & engineering
2. Train/test split (80/20, stratified on rating bins)
3. Preprocessing via ColumnTransformer
   - Numerical  → StandardScaler
   - Binary     → passthrough
   - Categorical → OneHotEncoder (handle_unknown='ignore')
4. Models trained
   - Linear Regression          (baseline)
   - Ridge Regression           (regularised linear)
   - Random Forest Regressor    (ensemble)
   - Gradient Boosting Regressor (boosted ensemble — best performer)
5. Evaluation: MAE, RMSE, R² Score
6. Best model selected and saved as a dict for re-use
"""

import os
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import StandardScaler, OneHotEncoder
from sklearn.compose           import ColumnTransformer
from sklearn.pipeline          import Pipeline
from sklearn.linear_model      import LinearRegression, Ridge
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics           import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Feature groups ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES     = ["Votes", "Average Cost for two"]
BINARY_FEATURES      = ["Has Online delivery", "Has Table booking"]
CATEGORICAL_FEATURES = ["Price range", "Primary Cuisine", "City"]
TARGET               = "Aggregate rating"

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


# ═══════════════════════════════════════════════════════════════════════════════
def prepare_features(df: pd.DataFrame):
    """
    Select and return X (features) and y (target) arrays.
    Drops rows with NaN in any used column.
    """
    cols_needed = ALL_FEATURES + [TARGET]
    sub = df[cols_needed].dropna().copy()

    # Ensure Price range is treated as categorical (object)
    sub["Price range"] = sub["Price range"].astype(str)

    X = sub[ALL_FEATURES]
    y = sub[TARGET]
    print(f"[features] Dataset for modelling: {X.shape[0]} rows × {X.shape[1]} features")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer preprocessing step."""
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  numeric_transformer,     NUMERIC_FEATURES),
            ("bin",  "passthrough",            BINARY_FEATURES),
            ("cat",  categorical_transformer,  CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


# ═══════════════════════════════════════════════════════════════════════════════
def build_models() -> dict:
    """Return a dict of model name → sklearn estimator."""
    return {
        "Linear Regression":       LinearRegression(),
        "Ridge Regression":        Ridge(alpha=1.0),
        "Random Forest":           RandomForestRegressor(
                                       n_estimators=200, max_depth=10,
                                       random_state=42, n_jobs=-1),
        "Gradient Boosting":       GradientBoostingRegressor(
                                       n_estimators=200, learning_rate=0.08,
                                       max_depth=5, random_state=42),
    }


# ═══════════════════════════════════════════════════════════════════════════════
def train_and_evaluate(df: pd.DataFrame) -> dict:
    """
    Full training & evaluation pipeline.

    Returns a dict with keys:
      - results      : DataFrame of model metrics
      - best_model   : fitted pipeline of the best model
      - best_name    : str name of the best model
      - X_test       : test features
      - y_test       : true test labels
      - y_pred_best  : predictions from the best model
      - feature_importance : DataFrame (for tree models)
      - preprocessor : the fitted ColumnTransformer
    """
    X, y = prepare_features(df)

    # ── Stratified split using rating bins ────────────────────────────────────
    bins = pd.cut(y, bins=[0, 2.5, 3.5, 4.0, 4.5, 5.0],
                  labels=["0-2.5", "2.5-3.5", "3.5-4.0", "4.0-4.5", "4.5-5.0"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=bins
    )
    print(f"[split]  Train: {len(X_train)}, Test: {len(X_test)}")

    preprocessor = build_preprocessor()
    models       = build_models()
    results      = []

    fitted_pipelines = {}
    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model",        model),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        results.append({"Model": name, "MAE": round(mae, 4),
                        "RMSE": round(rmse, 4), "R2 Score": round(r2, 4)})
        fitted_pipelines[name] = pipe
        print(f"  [{name}]  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False).reset_index(drop=True)
    print("\n[results]\n", results_df.to_string(index=False))

    # ── Best model ─────────────────────────────────────────────────────────────
    best_name  = results_df.iloc[0]["Model"]
    best_pipe  = fitted_pipelines[best_name]
    y_pred_best = best_pipe.predict(X_test)
    print(f"\n[best]   Best model: {best_name} (R²={results_df.iloc[0]['R2 Score']:.4f})")

    # ── Feature importance (tree models only) ──────────────────────────────────
    feature_importance_df = None
    best_model_obj = best_pipe.named_steps["model"]
    if hasattr(best_model_obj, "feature_importances_"):
        feature_importance_df = _extract_feature_importance(
            best_pipe, best_model_obj
        )

    return {
        "results":           results_df,
        "best_model":        best_pipe,
        "best_name":         best_name,
        "X_test":            X_test,
        "y_test":            y_test,
        "y_pred_best":       y_pred_best,
        "feature_importance": feature_importance_df,
        "fitted_pipelines":  fitted_pipelines,
        "X_train":           X_train,
        "y_train":           y_train,
    }


# ═══════════════════════════════════════════════════════════════════════════════
def _extract_feature_importance(pipeline: Pipeline, model) -> pd.DataFrame:
    """
    Extract feature names from ColumnTransformer and match with importances.
    """
    preprocessor = pipeline.named_steps["preprocessor"]

    # Numeric features
    num_names = NUMERIC_FEATURES.copy()
    # Binary features (passthrough)
    bin_names = BINARY_FEATURES.copy()
    # OHE categorical features
    ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))

    all_feature_names = num_names + bin_names + cat_names

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature":    all_feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return importance_df


# ═══════════════════════════════════════════════════════════════════════════════
def predict_single(pipeline: Pipeline, restaurant: dict) -> float:
    """
    Predict rating for a single restaurant dict.
    Keys must match ALL_FEATURES.
    """
    row = pd.DataFrame([restaurant])
    row["Price range"] = row["Price range"].astype(str)
    return float(pipeline.predict(row)[0])
