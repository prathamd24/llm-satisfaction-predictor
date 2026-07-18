"""
src/config.py
=============
Central configuration for the LLM Satisfaction Predictor project.

All paths, feature lists, model hyperparameters, and API settings are
defined here so the rest of the project has a single source of truth.

To override the dataset path, set the DATA_PATH environment variable.
"""

import os
from pathlib import Path

# ── Project root (one directory above this file) ──────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_PATH: Path = Path(
    os.getenv(
        "DATA_PATH",
        str(ROOT_DIR / "data" / "raw" / "genai_llm_usage_dataset_1000.csv"),
    )
)

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR:  Path = ROOT_DIR / "models"
MODEL_PATH:  Path = MODELS_DIR / "satisfaction_pipeline.pkl"

# ── Artifact paths ────────────────────────────────────────────────────────────
ARTIFACTS_DIR:          Path = ROOT_DIR / "artifacts"
METRICS_PATH:           Path = ARTIFACTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH: Path = ARTIFACTS_DIR / "feature_importance.csv"
MODEL_METADATA_PATH:    Path = ARTIFACTS_DIR / "model_metadata.json"
PLOTS_DIR:              Path = ARTIFACTS_DIR / "plots"

# ── Feature definitions ───────────────────────────────────────────────────────
# These are the features available BEFORE or DURING an LLM request.
#
# EXCLUDED due to leakage (only known AFTER the response):
#   - latency_sec        → measured after LLM responds
#   - hallucination_flag → evaluated after reading the response
#   - successful_response → determined after the response
#   - estimated_cost_usd → fully known only after usage
#   - total_tokens       → includes output/completion tokens (post-response)
#   - session_id         → row identifier, no predictive value
#   - user_satisfaction  → this IS the target, never an input

NUMERIC_FEATURES: list[str] = [
    "prompt_length",   # length of the prompt — known before sending
    "temperature",     # model parameter — set before the request
    "top_p",           # model parameter — set before the request
    "rag_enabled",     # binary 0/1 config flag — set before the request
]

CATEGORICAL_FEATURES: list[str] = [
    "model_name",           # chosen before the request
    "application_domain",   # business context — known before
    "task_type",            # type of task — known before
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Target variable
TARGET: str = "user_satisfaction"

# Valid range for clipping predictions (observed in dataset: 3, 4, 5)
TARGET_MIN: float = 3.0
TARGET_MAX: float = 5.0

# ── Model hyperparameters ─────────────────────────────────────────────────────
# Baseline XGBoost configuration — reasonable for ~1,000 samples
XGBOOST_PARAMS: dict = {
    "n_estimators":    300,
    "learning_rate":   0.05,
    "max_depth":       5,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "objective":       "reg:squarederror",
    "random_state":    42,
    "n_jobs":          -1,
}

# ── Train/test split ──────────────────────────────────────────────────────────
TEST_SIZE:    float = 0.2
RANDOM_STATE: int   = 42

# ── API settings ──────────────────────────────────────────────────────────────
API_HOST:    str = os.getenv("API_HOST",    "0.0.0.0")
API_PORT:    int = int(os.getenv("API_PORT", "8000"))
BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")
