# ================================================================
# models.py — Data shapes for all requests and responses
# ================================================================
# Pydantic models define exactly what data looks like going IN
# and coming OUT of our API.
# FastAPI uses these for automatic validation — if someone sends
# wrong data, they get a clear error instead of a crash.
# ================================================================

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ── Enums ───────────────────────────────────────────────────────

class Verdict(str, Enum):
    """Possible outcomes of our fake news detection."""
    FAKE      = "FAKE"        # Model is confident it's fake
    REAL      = "REAL"        # Model is confident it's real
    UNCERTAIN = "UNCERTAIN"   # Confidence below threshold


# ── Request Model ───────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """
    Shape of the request coming FROM Streamlit TO FastAPI.
    Streamlit sends this when user clicks 'Analyze'.
    """
    text: str = Field(
        ...,                          # ... means required, cannot be empty
        min_length=50,                # reject very short inputs
        max_length=10000,             # reject absurdly long inputs
        description="The news article text to analyze"
    )

    class Config:
        # Example shown in FastAPI's auto-generated docs at /docs
        json_schema_extra = {
            "example": {
                "text": "Scientists discover that drinking coffee "
                        "reverses aging by 20 years, study confirms."
            }
        }


# ── Response Models ─────────────────────────────────────────────

class BertResult(BaseModel):
    """
    Output from Stage 1 — the BERT model.
    This is an internal model, not directly returned to frontend.
    """
    verdict: Verdict              # FAKE / REAL / UNCERTAIN
    confidence: float             # 0.0 to 1.0 (e.g. 0.87 = 87%)
    fake_probability: float       # raw probability of being fake
    real_probability: float       # raw probability of being real


class LLMResult(BaseModel):
    """
    Output from Stage 2 — the LLM reasoning.
    Only generated when verdict is FAKE or UNCERTAIN.
    """
    reasoning: str                # Main explanation paragraph
    red_flags: list[str]          # List of manipulative techniques found
    what_to_verify: list[str]     # Suggested sources/steps to fact-check


class AnalyzeResponse(BaseModel):
    """
    Final shape of the response FROM FastAPI BACK TO Streamlit.
    This is everything the frontend needs to display results.
    """
    # ── Core result ──────────────────────────────────────────
    verdict: Verdict              # Final verdict shown to user
    confidence: float             # Confidence % shown in UI

    # ── Stage results ────────────────────────────────────────
    bert_result: BertResult       # Full BERT model output
    llm_result: Optional[LLMResult] = None
    # llm_result is Optional because:
    # - if verdict is REAL with high confidence, we skip LLM
    # - saves API cost + time for clearly real articles

    # ── Meta ─────────────────────────────────────────────────
    bert_model_used: str               # Which BERT model was used
    llm_provider_used: str        # Which LLM was used
    analysis_time_seconds: float  # How long the full analysis took