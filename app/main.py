"""
app/main.py
===========
FastAPI backend for the LLM Satisfaction Predictor.

Endpoints:
    GET  /          -> "LLM Satisfaction Predictor API is running"
    GET  /health    -> model loaded status
    POST /predict   -> return predicted satisfaction score

The ML pipeline is loaded ONCE at startup (lifespan context manager).
It is never reloaded per request — this keeps the API fast.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

API docs:
    http://localhost:8000/docs
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth import verify_firebase_token
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse, RootResponse
from app.services import get_prediction, is_model_loaded
from src.predict import load_pipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML pipeline once when the server starts."""
    logger.info("Starting LLM Satisfaction Predictor API...")
    try:
        load_pipeline()
        logger.info("Model pipeline loaded successfully.")
    except FileNotFoundError as exc:
        logger.error(str(exc))
        logger.error("Run `python src/train.py` to train the model first.")
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="LLM Satisfaction Predictor API",
    description=(
        "Predicts the expected **user satisfaction score** (3–5) for a given "
        "LLM usage configuration using an XGBoost regression model.\n\n"
        "Run `python src/train.py` before starting the API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=RootResponse, tags=["Status"])
def root() -> RootResponse:
    """Root endpoint — confirms the API is running."""
    return RootResponse(message="LLM Satisfaction Predictor API is running")


@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health() -> HealthResponse:
    """Health check — reports whether the ML model is loaded."""
    return HealthResponse(status="healthy", model_loaded=is_model_loaded())


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Predict user satisfaction score",
)
def predict_satisfaction(
    request: PredictionRequest,
    user: dict = Depends(verify_firebase_token)
) -> PredictionResponse:
    """
    Predict the expected user satisfaction score for an LLM usage session.

    Input: LLM configuration (model, domain, task, prompt settings).
    Output: Predicted satisfaction score in range [3, 5].

    Features used (all available BEFORE the LLM request):
    - model_name, application_domain, task_type
    - prompt_length, temperature, top_p, rag_enabled

    Features intentionally excluded (post-response leakage):
    - latency_sec, hallucination_flag, successful_response,
      estimated_cost_usd, total_tokens
    """
    try:
        result = get_prediction(request.model_dump())
        return PredictionResponse(
            predicted_satisfaction=result["predicted_satisfaction"],
            scale="3-5",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected prediction error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred during prediction: {str(exc)}",
        )
