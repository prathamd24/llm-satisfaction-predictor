"""
app/schemas.py
==============
Pydantic models for the LLM Satisfaction Predictor API.

Pydantic automatically:
  - Validates that required fields are present
  - Validates field types
  - Returns a clear 422 error if validation fails

This means we don't need manual validation in the route handlers.
"""

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input schema for POST /predict.

    These are the 7 features available BEFORE an LLM request is made.
    total_tokens, latency_sec, hallucination_flag, successful_response,
    and estimated_cost_usd are intentionally excluded (post-response leakage).
    """

    model_name: Literal[
        "GPT-4o",
        "Claude 3.7",
        "Gemma 3",
        "Llama 3.1",
        "Mistral Large",
        "Qwen 2.5",
    ] = Field(..., description="LLM model used for the request.")

    application_domain: Literal[
        "Coding",
        "Customer Support",
        "Education",
        "Finance",
        "Healthcare",
        "Legal",
        "Retail",
    ] = Field(..., description="Business domain of the application.")

    task_type: Literal[
        "Classification",
        "Code Generation",
        "QA",
        "RAG",
        "Summarization",
        "Translation",
    ] = Field(..., description="Type of task the LLM is performing.")

    prompt_length: int = Field(
        ..., gt=0, le=5000,
        description="Number of characters/tokens in the prompt (before sending)."
    )

    temperature: float = Field(
        ..., ge=0.0, le=2.0,
        description="LLM temperature parameter (controls randomness, 0.0-2.0)."
    )

    top_p: float = Field(
        ..., ge=0.0, le=1.0,
        description="LLM top-p (nucleus sampling) parameter (0.0-1.0)."
    )

    rag_enabled: Literal[0, 1] = Field(
        ..., description="Whether Retrieval-Augmented Generation is enabled (0 or 1)."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_name":         "GPT-4o",
                "application_domain": "Education",
                "task_type":          "QA",
                "prompt_length":      800,
                "temperature":        0.7,
                "top_p":              0.9,
                "rag_enabled":        1,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for POST /predict."""

    predicted_satisfaction: float = Field(
        ...,
        description="Predicted user satisfaction score, clipped to [3, 5]."
    )
    scale: str = Field(
        default="3-5",
        description="The satisfaction scale used in the training dataset."
    )


class HealthResponse(BaseModel):
    """Response schema for GET /health."""
    status: str
    model_loaded: bool


class RootResponse(BaseModel):
    """Response schema for GET /."""
    message: str
