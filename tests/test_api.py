"""
tests/test_api.py
=================
Integration tests for the LLM Satisfaction Predictor FastAPI backend.

Run:
    python -m pytest tests/test_api.py -v

All tests are skipped if the model pipeline has not been trained yet.
Run `python src/train.py` first.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_PATH

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model not trained yet. Run `python src/train.py` first."
)

VALID_PAYLOAD = {
    "model_name":         "GPT-4o",
    "application_domain": "Education",
    "task_type":          "QA",
    "prompt_length":      800,
    "temperature":        0.7,
    "top_p":              0.9,
    "rag_enabled":        1,
}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


# ── GET / ─────────────────────────────────────────────────────────────────────

class TestRoot:
    def test_status_200(self, client):
        assert client.get("/").status_code == 200

    def test_message_field(self, client):
        assert "message" in client.get("/").json()

    def test_message_content(self, client):
        msg = client.get("/").json()["message"]
        assert "LLM" in msg or "Satisfaction" in msg


# ── GET /health ───────────────────────────────────────────────────────────────

class TestHealth:
    def test_status_200(self, client):
        assert client.get("/health").status_code == 200

    def test_status_healthy(self, client):
        assert client.get("/health").json()["status"] == "healthy"

    def test_model_loaded_true(self, client):
        assert client.get("/health").json()["model_loaded"] is True


# ── POST /predict — happy paths ───────────────────────────────────────────────

class TestPredictHappyPath:
    def test_status_200(self, client):
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 200

    def test_response_has_predicted_satisfaction(self, client):
        assert "predicted_satisfaction" in client.post("/predict", json=VALID_PAYLOAD).json()

    def test_response_has_scale(self, client):
        assert "scale" in client.post("/predict", json=VALID_PAYLOAD).json()

    def test_scale_is_3_5(self, client):
        assert client.post("/predict", json=VALID_PAYLOAD).json()["scale"] == "3-5"

    def test_prediction_is_float(self, client):
        score = client.post("/predict", json=VALID_PAYLOAD).json()["predicted_satisfaction"]
        assert isinstance(score, float)

    def test_prediction_in_valid_range(self, client):
        score = client.post("/predict", json=VALID_PAYLOAD).json()["predicted_satisfaction"]
        assert 3.0 <= score <= 5.0, f"Score {score} outside [3, 5]"

    def test_claude_model(self, client):
        payload = {**VALID_PAYLOAD, "model_name": "Claude 3.7"}
        assert client.post("/predict", json=payload).status_code == 200

    def test_rag_disabled(self, client):
        payload = {**VALID_PAYLOAD, "rag_enabled": 0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert 3.0 <= resp.json()["predicted_satisfaction"] <= 5.0

    def test_healthcare_domain(self, client):
        payload = {**VALID_PAYLOAD, "application_domain": "Healthcare"}
        assert client.post("/predict", json=payload).status_code == 200


# ── POST /predict — validation errors ────────────────────────────────────────

class TestPredictValidationErrors:
    def test_missing_model_name(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "model_name"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_missing_task_type(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "task_type"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_invalid_model_name(self, client):
        assert client.post("/predict", json={**VALID_PAYLOAD, "model_name": "FakeAI"}).status_code == 422

    def test_invalid_domain(self, client):
        assert client.post("/predict", json={**VALID_PAYLOAD, "application_domain": "Space"}).status_code == 422

    def test_invalid_task_type(self, client):
        assert client.post("/predict", json={**VALID_PAYLOAD, "task_type": "Dreaming"}).status_code == 422

    def test_negative_prompt_length(self, client):
        assert client.post("/predict", json={**VALID_PAYLOAD, "prompt_length": -5}).status_code == 422

    def test_temperature_too_high(self, client):
        assert client.post("/predict", json={**VALID_PAYLOAD, "temperature": 3.0}).status_code == 422

    def test_empty_body(self, client):
        assert client.post("/predict", json={}).status_code == 422
