"""
src/predict.py
==============
Prediction service for the LLM Satisfaction Predictor.

The pipeline is loaded ONCE (using lru_cache) and reused for every call.
This is efficient and important for the FastAPI backend — we don't want to
reload the model from disk on every HTTP request.

Usage example:
    from src.predict import predict

    result = predict({
        "model_name":         "GPT-4o",
        "application_domain": "Education",
        "task_type":          "QA",
        "prompt_length":      800,
        "temperature":        0.7,
        "top_p":              0.9,
        "rag_enabled":        1,
    })
    # Returns: {"predicted_satisfaction": 4.25}
"""

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALL_FEATURES, MODEL_PATH, TARGET_MAX, TARGET_MIN

logger = logging.getLogger(__name__)

# ── Valid vocabulary for categorical fields ───────────────────────────────────
# These match the unique values observed in the training data.
# Unseen values are handled by OneHotEncoder(handle_unknown='ignore'),
# but we still validate them for clear error messages.
VALID_MODEL_NAMES = {
    "Claude 3.7", "GPT-4o", "Gemma 3",
    "Llama 3.1", "Mistral Large", "Qwen 2.5",
}
VALID_DOMAINS = {
    "Coding", "Customer Support", "Education",
    "Finance", "Healthcare", "Legal", "Retail",
}
VALID_TASK_TYPES = {
    "Classification", "Code Generation", "QA",
    "RAG", "Summarization", "Translation",
}


@lru_cache(maxsize=1)
def load_pipeline():
    """
    Load the trained sklearn Pipeline from disk.

    lru_cache means this function only reads the file once per Python process,
    no matter how many times it is called. Subsequent calls return the cached object.

    Raises:
        FileNotFoundError: If satisfaction_pipeline.pkl does not exist.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'.\n"
            "Run `python src/train.py` to train and save the model first."
        )
    pipeline = joblib.load(MODEL_PATH)
    logger.info(f"Pipeline loaded from {MODEL_PATH}")
    return pipeline


def validate_input(data: dict[str, Any]) -> None:
    """
    Check that all required fields are present and contain valid values.

    Raises:
        ValueError: With a descriptive message for the first problem found.
    """
    # Check all required fields are present
    missing = [f for f in ALL_FEATURES if f not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Validate categorical values
    if data["model_name"] not in VALID_MODEL_NAMES:
        raise ValueError(
            f"'model_name' must be one of {sorted(VALID_MODEL_NAMES)}. "
            f"Got: {data['model_name']!r}"
        )
    if data["application_domain"] not in VALID_DOMAINS:
        raise ValueError(
            f"'application_domain' must be one of {sorted(VALID_DOMAINS)}. "
            f"Got: {data['application_domain']!r}"
        )
    if data["task_type"] not in VALID_TASK_TYPES:
        raise ValueError(
            f"'task_type' must be one of {sorted(VALID_TASK_TYPES)}. "
            f"Got: {data['task_type']!r}"
        )

    # Validate numeric ranges
    prompt_length = data["prompt_length"]
    if not isinstance(prompt_length, (int, float)) or prompt_length <= 0:
        raise ValueError(
            f"'prompt_length' must be a positive number. Got: {prompt_length!r}"
        )

    temp = data["temperature"]
    if not isinstance(temp, (int, float)) or not (0.0 <= temp <= 2.0):
        raise ValueError(
            f"'temperature' must be between 0.0 and 2.0. Got: {temp!r}"
        )

    top_p = data["top_p"]
    if not isinstance(top_p, (int, float)) or not (0.0 <= top_p <= 1.0):
        raise ValueError(
            f"'top_p' must be between 0.0 and 1.0. Got: {top_p!r}"
        )

    rag = data["rag_enabled"]
    if rag not in (0, 1, True, False):
        raise ValueError(
            f"'rag_enabled' must be 0 or 1. Got: {rag!r}"
        )


def predict(input_data: dict[str, Any]) -> dict[str, float]:
    """
    Run the full preprocessing + XGBoost pipeline on one input dictionary.

    Steps:
      1. Validate all input fields.
      2. Load the pipeline (from cache after first call).
      3. Build a single-row DataFrame.
      4. Run pipeline.predict().
      5. Clip to [TARGET_MIN, TARGET_MAX] = [3.0, 5.0].
      6. Return rounded prediction.

    Args:
        input_data: dict with keys matching ALL_FEATURES.

    Returns:
        {"predicted_satisfaction": float}  (clipped and rounded to 2 dp)
    """
    validate_input(input_data)

    pipeline = load_pipeline()

    # Build single-row DataFrame in the exact column order the pipeline expects
    df = pd.DataFrame([input_data])[ALL_FEATURES]

    raw_prediction = float(pipeline.predict(df)[0])

    # Clip to valid range — the model might predict slightly outside [3, 5]
    clipped = float(np.clip(raw_prediction, TARGET_MIN, TARGET_MAX))

    logger.debug(f"Raw: {raw_prediction:.4f} -> Clipped: {clipped:.4f}")
    return {"predicted_satisfaction": round(clipped, 2)}
