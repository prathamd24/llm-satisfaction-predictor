"""
tests/test_model.py
===================
Unit tests for the LLM Satisfaction Predictor model pipeline.

Run:
    python -m pytest tests/test_model.py -v

Tests that require the trained model file are automatically skipped
if models/satisfaction_pipeline.pkl does not exist yet.
Run `python src/train.py` first.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_PATH, TARGET_MAX, TARGET_MIN

SAMPLE_INPUT = {
    "model_name":         "GPT-4o",
    "application_domain": "Education",
    "task_type":          "QA",
    "prompt_length":      800,
    "temperature":        0.7,
    "top_p":              0.9,
    "rag_enabled":        1,
}


# ── Tests that require the trained model ──────────────────────────────────────

@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model not trained yet. Run `python src/train.py` first."
)
class TestModelPipeline:

    def test_model_file_exists(self):
        assert MODEL_PATH.exists(), f"Expected model at {MODEL_PATH}"

    def test_pipeline_loads(self):
        from src.predict import load_pipeline
        assert load_pipeline() is not None

    def test_pipeline_has_steps(self):
        from src.predict import load_pipeline
        pipeline = load_pipeline()
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_prediction_returns_dict(self):
        from src.predict import predict
        result = predict(SAMPLE_INPUT)
        assert isinstance(result, dict)
        assert "predicted_satisfaction" in result

    def test_prediction_is_float(self):
        from src.predict import predict
        result = predict(SAMPLE_INPUT)
        assert isinstance(result["predicted_satisfaction"], float)

    def test_prediction_is_not_nan(self):
        from src.predict import predict
        result = predict(SAMPLE_INPUT)
        assert not np.isnan(result["predicted_satisfaction"])

    def test_prediction_within_valid_range(self):
        from src.predict import predict
        result = predict(SAMPLE_INPUT)
        score = result["predicted_satisfaction"]
        assert TARGET_MIN <= score <= TARGET_MAX, (
            f"Prediction {score} outside expected range [{TARGET_MIN}, {TARGET_MAX}]"
        )

    def test_different_models_produce_results(self):
        from src.predict import predict
        for model in ["GPT-4o", "Claude 3.7", "Llama 3.1"]:
            r = predict({**SAMPLE_INPUT, "model_name": model})
            assert TARGET_MIN <= r["predicted_satisfaction"] <= TARGET_MAX

    def test_rag_disabled_still_predicts(self):
        from src.predict import predict
        r = predict({**SAMPLE_INPUT, "rag_enabled": 0})
        assert "predicted_satisfaction" in r


# ── Input validation tests (no model required) ────────────────────────────────

class TestInputValidation:

    def test_missing_field_raises(self):
        from src.predict import validate_input
        incomplete = {k: v for k, v in SAMPLE_INPUT.items() if k != "model_name"}
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_input(incomplete)

    def test_invalid_model_name_raises(self):
        from src.predict import validate_input
        bad = {**SAMPLE_INPUT, "model_name": "FakeModel-9000"}
        with pytest.raises(ValueError, match="model_name"):
            validate_input(bad)

    def test_invalid_domain_raises(self):
        from src.predict import validate_input
        bad = {**SAMPLE_INPUT, "application_domain": "Space"}
        with pytest.raises(ValueError, match="application_domain"):
            validate_input(bad)

    def test_invalid_task_type_raises(self):
        from src.predict import validate_input
        bad = {**SAMPLE_INPUT, "task_type": "Meditation"}
        with pytest.raises(ValueError, match="task_type"):
            validate_input(bad)

    def test_negative_prompt_length_raises(self):
        from src.predict import validate_input
        bad = {**SAMPLE_INPUT, "prompt_length": -10}
        with pytest.raises(ValueError, match="prompt_length"):
            validate_input(bad)

    def test_temperature_out_of_range_raises(self):
        from src.predict import validate_input
        bad = {**SAMPLE_INPUT, "temperature": 3.0}
        with pytest.raises(ValueError, match="temperature"):
            validate_input(bad)

    def test_top_p_out_of_range_raises(self):
        from src.predict import validate_input
        bad = {**SAMPLE_INPUT, "top_p": 1.5}
        with pytest.raises(ValueError, match="top_p"):
            validate_input(bad)

    def test_all_valid_domains_pass(self):
        from src.predict import validate_input
        for domain in ["Coding", "Customer Support", "Education",
                       "Finance", "Healthcare", "Legal", "Retail"]:
            validate_input({**SAMPLE_INPUT, "application_domain": domain})

    def test_all_valid_task_types_pass(self):
        from src.predict import validate_input
        for task in ["Classification", "Code Generation", "QA",
                     "RAG", "Summarization", "Translation"]:
            validate_input({**SAMPLE_INPUT, "task_type": task})
