"""
src/train.py
============
Training script for the LLM Satisfaction Predictor.

What this script does (step by step):
  1.  Load the dataset (genai_llm_usage_dataset_1000.csv)
  2.  Clean data — remove missing targets, drop duplicates
  3.  Select the 7 leakage-free input features + target
  4.  Split 80% train / 20% test (random_state=42 for reproducibility)
  5.  Build a sklearn Pipeline: ColumnTransformer -> XGBRegressor
  6.  Train the pipeline on training data ONLY
  7.  Predict on test data
  8.  Evaluate: MAE, RMSE, R2
  9.  Save the fitted pipeline -> models/satisfaction_pipeline.pkl
  10. Save metrics -> artifacts/metrics.json
  11. Save feature importance -> artifacts/feature_importance.csv
  12. Save evaluation plots -> artifacts/plots/
  13. Save model metadata -> artifacts/model_metadata.json

Run with:
    python src/train.py
"""

import datetime
import logging
import platform
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Make sure Python can find the src/ package when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

matplotlib.use("Agg")  # non-interactive — safe for scripts and servers
import matplotlib.pyplot as plt

from src.config import (
    ALL_FEATURES,
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PLOTS_DIR,
    RANDOM_STATE,
    TARGET,
    TARGET_MAX,
    TARGET_MIN,
    TEST_SIZE,
    XGBOOST_PARAMS,
)
from src.data_processing import build_preprocessor, clean_data, load_raw_data
from src.utils import ensure_dirs, save_json, setup_logging


def train() -> None:
    """Run the full training pipeline from raw data to saved artifacts."""
    setup_logging()
    logger = logging.getLogger(__name__)

    ensure_dirs([MODELS_DIR, ARTIFACTS_DIR, PLOTS_DIR])

    # ── 1. Load & clean ───────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("PHASE 1: Loading and cleaning data")
    logger.info("=" * 55)
    df = load_raw_data()
    df = clean_data(df)

    # ── 2. Feature / target selection ────────────────────────────────────────
    logger.info("PHASE 2: Selecting features and target")
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    logger.info(f"Features ({len(ALL_FEATURES)}): {ALL_FEATURES}")
    logger.info(f"Target: {TARGET} | Values: {sorted(y.unique())} | Samples: {len(y):,}")

    # ── 3. Train / test split ─────────────────────────────────────────────────
    logger.info("PHASE 3: Train/test split (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── 4. Build pipeline ─────────────────────────────────────────────────────
    logger.info("PHASE 4: Building pipeline")
    # The Pipeline ensures the preprocessor is fit ONLY on training data.
    # When pipeline.fit(X_train, y_train) is called, sklearn automatically:
    #   a) Fits the ColumnTransformer on X_train
    #   b) Transforms X_train
    #   c) Fits XGBRegressor on the transformed data
    preprocessor = build_preprocessor()
    xgb_model = XGBRegressor(**XGBOOST_PARAMS)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model",        xgb_model),
    ])

    # ── 5. Train ──────────────────────────────────────────────────────────────
    logger.info("PHASE 5: Training XGBoost (this may take a moment)...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    logger.info("PHASE 6: Evaluating on held-out test set")
    y_pred_raw = pipeline.predict(X_test)
    # Clip predictions to the valid target range [3, 5]
    y_pred = np.clip(y_pred_raw, TARGET_MIN, TARGET_MAX)

    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))

    logger.info("-" * 40)
    logger.info(f"  MAE  : {mae:.4f}  (avg error in satisfaction points)")
    logger.info(f"  RMSE : {rmse:.4f} (penalises large errors more)")
    logger.info(f"  R2   : {r2:.4f}  (1.0 = perfect, 0.0 = predicts mean)")
    logger.info("-" * 40)

    # NOTE: With only 1,000 samples and a target that only takes 3 values
    # (3, 4, 5), R2 may be low. This is expected and does not indicate a bug.
    # More data and richer features would improve performance.

    # ── 7. Save pipeline ──────────────────────────────────────────────────────
    logger.info("PHASE 7: Saving pipeline")
    joblib.dump(pipeline, MODEL_PATH)
    logger.info(f"Pipeline saved -> {MODEL_PATH}")

    # ── 8. Save metrics ───────────────────────────────────────────────────────
    metrics = {
        "mae":  round(mae,  4),
        "rmse": round(rmse, 4),
        "r2":   round(r2,   4),
    }
    save_json(metrics, METRICS_PATH)
    logger.info(f"Metrics saved  -> {METRICS_PATH}")

    # ── 9. Feature importance ─────────────────────────────────────────────────
    logger.info("PHASE 9: Saving feature importance")
    _save_feature_importance(pipeline, logger)

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    logger.info("PHASE 10: Generating plots")
    _save_plots(y_test, y_pred)
    logger.info(f"Plots saved    -> {PLOTS_DIR}")

    # ── 11. Model metadata ────────────────────────────────────────────────────
    logger.info("PHASE 11: Saving model metadata")
    metadata = {
        "model_name":     "XGBoostRegressor",
        "project":        "LLM Satisfaction Predictor",
        "training_date":  datetime.datetime.now().isoformat(),
        "dataset_path":   str(DATA_PATH),
        "dataset_rows":   len(df),
        "features":       ALL_FEATURES,
        "target":         TARGET,
        "target_range":   f"{TARGET_MIN}-{TARGET_MAX}",
        "model_params":   XGBOOST_PARAMS,
        "test_size":      TEST_SIZE,
        "random_state":   RANDOM_STATE,
        "mae":            round(mae,  4),
        "rmse":           round(rmse, 4),
        "r2":             round(r2,   4),
        "python_version": platform.python_version(),
    }
    save_json(metadata, MODEL_METADATA_PATH)
    logger.info(f"Metadata saved -> {MODEL_METADATA_PATH}")

    logger.info("=" * 55)
    logger.info("Training complete. All artifacts saved.")
    logger.info("=" * 55)


def _save_feature_importance(pipeline: Pipeline, logger: logging.Logger) -> None:
    """Extract XGBoost feature importances and save as CSV + bar chart."""
    try:
        xgb_model    = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        cat_names = list(
            preprocessor.named_transformers_["cat"].get_feature_names_out(
                CATEGORICAL_FEATURES
            )
        )
        all_feature_names = NUMERIC_FEATURES + cat_names

        importances = xgb_model.feature_importances_
        fi_df = (
            pd.DataFrame({"feature": all_feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        logger.info(f"Feature importance -> {FEATURE_IMPORTANCE_PATH}")

        # Bar chart
        plt.figure(figsize=(10, 6))
        colors = ["#4F8EF7" if i < 5 else "#94a3b8" for i in range(len(fi_df))]
        plt.barh(fi_df["feature"], fi_df["importance"], color=colors)
        plt.xlabel("Importance (gain)")
        plt.title("Feature Importances — XGBoost LLM Satisfaction Model")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
        plt.close()

    except Exception as exc:
        logger.warning(f"Could not save feature importance: {exc}")


def _save_plots(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """Save Actual vs Predicted scatter and Residuals histogram."""
    # Actual vs Predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, s=25, color="#4F8EF7",
                label="Predictions")
    lo, hi = TARGET_MIN - 0.3, TARGET_MAX + 0.3
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Actual Satisfaction Score")
    plt.ylabel("Predicted Satisfaction Score")
    plt.title("Actual vs Predicted — User Satisfaction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Residuals distribution
    residuals = y_test.values - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=40, color="#4F8EF7", edgecolor="white", linewidth=0.4)
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title("Residual Distribution — User Satisfaction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residuals.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    train()
