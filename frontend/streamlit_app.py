"""
frontend/streamlit_app.py
=========================
LLM Satisfaction Predictor — Streamlit Frontend.

This UI collects LLM usage configuration from the user,
sends it to the FastAPI backend, and displays the predicted
user satisfaction score.

The ML model lives ONLY in the backend.
This file never performs any prediction itself.

Run:
    streamlit run frontend/streamlit_app.py
"""

import os

import requests
import streamlit as st

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Satisfaction Predictor",
    page_icon="🤖",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0f1117; }

.result-box {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin: 20px 0;
}
.result-label {
    color: #94a3b8;
    font-size: 0.9em;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.result-score {
    font-size: 3.5em;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.result-scale {
    color: #475569;
    font-size: 0.85em;
    margin-top: 8px;
}
.section-label {
    color: #64748b;
    font-size: 0.75em;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Backend health check ──────────────────────────────────────────────────────
def _check_backend() -> tuple[bool, bool]:
    """Return (reachable, model_loaded)."""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if r.status_code == 200:
            return True, r.json().get("model_loaded", False)
        return True, False
    except Exception:
        return False, False


backend_ok, model_ok = _check_backend()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 LLM Predictor")
    st.markdown("---")
    if backend_ok and model_ok:
        st.success("Backend connected\nModel ready")
    elif backend_ok:
        st.warning("Backend connected\nModel not loaded\nRun: `python src/train.py`")
    else:
        st.error(f"Cannot reach backend\n`{BACKEND_URL}`\n\nStart: `uvicorn app.main:app --reload`")

    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "Predicts the expected user satisfaction score "
        "for an LLM usage session using an XGBoost "
        "regression model trained on ~1,000 records."
    )
    st.markdown(f"Backend: `{BACKEND_URL}`")
    if backend_ok:
        st.markdown(f"[API Docs]({BACKEND_URL}/docs)")


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0 4px;'>
    <div style='font-size:2.6em; font-weight:700;
                background: linear-gradient(135deg,#60a5fa,#a78bfa);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                background-clip:text;'>
        🤖 LLM Satisfaction Predictor
    </div>
    <p style='color:#64748b; font-size:1em; margin:8px 0 0;'>
        Enter your LLM configuration to predict the expected user satisfaction score.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Section 1: LLM Usage Information ─────────────────────────────────────────
st.markdown('<div class="section-label">LLM Usage Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    model_name = st.selectbox(
        "Model",
        options=["GPT-4o", "Claude 3.7", "Gemma 3", "Llama 3.1", "Mistral Large", "Qwen 2.5"],
        help="The LLM model being used.",
    )
    application_domain = st.selectbox(
        "Application Domain",
        options=["Coding", "Customer Support", "Education", "Finance", "Healthcare", "Legal", "Retail"],
        help="The business domain of the application.",
    )
with col2:
    task_type = st.selectbox(
        "Task Type",
        options=["Classification", "Code Generation", "QA", "RAG", "Summarization", "Translation"],
        help="The type of task the LLM is performing.",
    )
    rag_enabled = st.selectbox(
        "RAG Enabled",
        options=[0, 1],
        format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
        help="Is Retrieval-Augmented Generation enabled?",
    )

# ── Section 2: Request Configuration ─────────────────────────────────────────
st.markdown('<div class="section-label">Request Configuration</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)
with col3:
    prompt_length = st.number_input(
        "Prompt Length",
        min_value=1,
        max_value=5000,
        value=800,
        step=50,
        help="Number of characters in the prompt (known before sending).",
    )
with col4:
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.05,
        help="Controls randomness (0 = deterministic, 2 = very random).",
    )
with col5:
    top_p = st.slider(
        "Top-p",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Nucleus sampling parameter.",
    )

st.markdown("---")

# ── Predict button ────────────────────────────────────────────────────────────
predict_clicked = st.button(
    "Predict Satisfaction",
    use_container_width=True,
    type="primary",
    disabled=(not backend_ok or not model_ok),
)

if not backend_ok:
    st.warning("Backend is not reachable. Start FastAPI first.")
elif not model_ok:
    st.warning("Model not loaded. Run `python src/train.py` and restart the API.")

# ── Prediction result ─────────────────────────────────────────────────────────
if predict_clicked and backend_ok and model_ok:
    payload = {
        "model_name":         model_name,
        "application_domain": application_domain,
        "task_type":          task_type,
        "prompt_length":      prompt_length,
        "temperature":        temperature,
        "top_p":              top_p,
        "rag_enabled":        rag_enabled,
    }

    with st.spinner("Predicting..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict",
                json=payload,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                score = data["predicted_satisfaction"]
                scale = data.get("scale", "3-5")

                # Build star display
                full_stars  = int(score - 3) + 1      # rough star count
                stars_str   = "⭐" * max(1, min(full_stars, 3))

                st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Predicted User Satisfaction</div>
                        <div class="result-score">{stars_str} {score:.2f}</div>
                        <div class="result-scale">Scale: {scale}</div>
                    </div>
                """, unsafe_allow_html=True)

                # Simple interpretation
                if score >= 4.5:
                    st.success("Very high satisfaction expected.")
                elif score >= 4.0:
                    st.info("High satisfaction expected.")
                elif score >= 3.5:
                    st.warning("Moderate satisfaction expected.")
                else:
                    st.error("Low satisfaction expected.")

            elif response.status_code == 422:
                st.error(f"Validation error: {response.json().get('detail', 'Unknown')}")
            elif response.status_code == 503:
                st.error("Model not available. Run `python src/train.py` first.")
            else:
                st.error(f"API error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to backend at `{BACKEND_URL}`.")
        except requests.exceptions.Timeout:
            st.error("Request timed out.")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.75em; padding-top:40px;'>
LLM Satisfaction Predictor &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; FastAPI + Streamlit
</div>
""", unsafe_allow_html=True)
