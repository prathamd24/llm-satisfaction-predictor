"""
src/data_processing.py
=======================
Reusable data loading and preprocessing for the LLM Satisfaction Predictor.

KEY DESIGN DECISIONS
--------------------
1. The ColumnTransformer is fit ONLY on training data — never on the full dataset.
   This prevents data leakage between train and test sets.

2. The following columns are EXCLUDED (post-response leakage):
   - latency_sec        : only measured after the LLM responds
   - hallucination_flag : only evaluated after reading the response
   - successful_response: only determined after the response arrives
   - estimated_cost_usd : fully known only after usage
   - total_tokens       : includes completion tokens (post-response)
   - session_id         : row identifier, no predictive value
   - user_satisfaction  : this IS the target variable

3. rag_enabled is 0/1 integer — treated as numeric (no encoding needed).
"""

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    NUMERIC_FEATURES,
    TARGET,
)

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    """
    Load the LLM usage CSV from the configured DATA_PATH.

    Raises:
        FileNotFoundError: If the CSV is not found at DATA_PATH.
    """
    logger.info(f"Loading dataset from: {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'.\n"
            "Please copy 'genai_llm_usage_dataset_1000.csv' into 'data/raw/' "
            "or set the DATA_PATH environment variable."
        )
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df):,} rows x {len(df.columns)} columns.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw LLM usage DataFrame.

    Steps:
      1. Drop rows where the target (user_satisfaction) is missing.
      2. Drop duplicate rows.
      3. Keep only the columns needed for modelling.

    Note: This dataset has no missing values, but we handle it gracefully anyway.

    Args:
        df: Raw DataFrame from load_raw_data().

    Returns:
        Cleaned DataFrame with only the feature columns + target column.
    """
    df = df.copy()
    before = len(df)

    # Drop rows where the target is missing
    df = df.dropna(subset=[TARGET])
    dropped_target = before - len(df)
    if dropped_target > 0:
        logger.info(f"Dropped {dropped_target} rows with missing target.")

    # Drop duplicate rows
    df = df.drop_duplicates()
    dropped_dup = before - len(df) - dropped_target
    if dropped_dup > 0:
        logger.info(f"Dropped {dropped_dup} duplicate rows.")

    # Keep only the columns we actually need
    keep_cols = ALL_FEATURES + [TARGET]
    df = df[keep_cols].copy()

    logger.info(f"After cleaning: {len(df):,} rows remaining.")
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build the sklearn ColumnTransformer for the 7 selected features.

    Transformations:
      - Numeric features (prompt_length, temperature, top_p, rag_enabled):
        StandardScaler — centres and scales each feature.
        (Tree-based models like XGBoost don't require scaling, but it adds
        consistency when switching models later.)

      - Categorical features (model_name, application_domain, task_type):
        OneHotEncoder — converts each category into binary columns.
        handle_unknown='ignore' means unseen categories at prediction time
        become all-zeros (safe fallback, no crash).

    IMPORTANT: This preprocessor must be fit ONLY on X_train.
               In train.py it is wrapped inside a sklearn Pipeline so
               sklearn handles this automatically.

    Returns:
        An unfitted ColumnTransformer instance.
    """
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,  # return dense array (sklearn >= 1.2)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # silently ignore any other columns
    )
    return preprocessor
