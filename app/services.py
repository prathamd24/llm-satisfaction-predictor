"""
app/services.py
===============
Business logic layer between the FastAPI routes and the prediction module.

Keeping this layer lets us test the prediction logic independently of HTTP.
"""

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import load_pipeline, predict

logger = logging.getLogger(__name__)


def get_prediction(input_data: dict[str, Any]) -> dict[str, float]:
    """Call the prediction service and return the result dict."""
    return predict(input_data)


def is_model_loaded() -> bool:
    """Return True if the pipeline is loaded in memory."""
    try:
        load_pipeline()
        return True
    except Exception as exc:
        logger.warning(f"Model not loaded: {exc}")
        return False
