# ================================================================
# main.py — FastAPI server, all API routes live here
# ================================================================
# This file:
#   1. Creates the FastAPI app
#   2. Loads the BERT model on startup (so first request is fast)
#   3. Exposes two endpoints:
#      GET  /health  → check if server is running
#      POST /analyze → main analysis endpoint
#
# To run this server:
#   uvicorn backend.main:app --reload --port 8000
# ================================================================

import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.models import AnalyzeRequest, AnalyzeResponse
from backend.analyzer import analyze_news
from backend.ml.bert_detector import detector
from backend.config import validate_config, BERT_MODEL_NAME, LLM_PROVIDER

# ── Logging Setup ────────────────────────────────────────────────
# This makes all log messages show up nicely in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ── Startup & Shutdown ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on server startup and shutdown.
    We use startup to:
      1. Validate config (catch missing API keys early)
      2. Pre-load the BERT model (so first request isn't slow)
    """
    # ── STARTUP ──────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Starting Fake News Detector API...")
    logger.info(f"BERT Model  : {BERT_MODEL_NAME}")
    logger.info(f"LLM Provider: {LLM_PROVIDER}")
    logger.info("=" * 50)

    # Step 1: Validate config — crash early if keys are missing
    try:
        validate_config()
        logger.info("Config validation passed ✅")
    except ValueError as e:
        logger.error(f"Config validation failed: {e}")
        raise  # Stop server from starting with bad config

    # Step 2: Pre-load BERT model so first request is fast
    # Without this, the first user request would wait ~10s for model load
    logger.info("Pre-loading BERT model (this may take a moment)...")
    try:
        detector.load_model()
        logger.info("BERT model loaded and ready ✅")
    except Exception as e:
        logger.error(f"Failed to pre-load BERT model: {e}")
        raise  # Can't run without the ML model

    logger.info("Server ready! Listening for requests...")
    logger.info("=" * 50)

    yield  # Server runs here

    # ── SHUTDOWN ─────────────────────────────────────────────
    logger.info("Shutting down server...")


# ── App Instance ─────────────────────────────────────────────────
app = FastAPI(
    title="Fake News Detector API",
    description=(
        "2-stage fake news detection: "
        "BERT model for verdict + LLM for reasoning"
    ),
    version="1.0.0",
    lifespan=lifespan      # attach our startup/shutdown logic
)


# ── CORS Middleware ──────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing
# This allows Streamlit (running on port 8501) to call
# our FastAPI server (running on port 8000)
# Without this, the browser would block the requests!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production, replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    Streamlit calls this to verify the backend is running.
    Also useful for deployment platforms to check server status.
    """
    return {
        "status": "healthy",
        "bert_model": BERT_MODEL_NAME,
        "bert_loaded": detector.is_loaded,
        "llm_provider": LLM_PROVIDER,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Main analysis endpoint.

    Receives news text from Streamlit,
    runs it through the 2-stage pipeline,
    returns full analysis result.

    FastAPI automatically:
      - Validates the request using AnalyzeRequest schema
      - Returns 422 error if text is too short/long
      - Serializes response using AnalyzeResponse schema
    """
    logger.info(
        f"Received analyze request "
        f"(text length: {len(request.text)} chars)"
    )

    try:
        # Run the full 2-stage analysis
        result = analyze_news(request.text)

        logger.info(
            f"Analysis complete: {result.verdict} "
            f"({result.confidence:.1%}) in {result.analysis_time_seconds}s"
        )

        return result

    except RuntimeError as e:
        # RuntimeError = something we expected might go wrong
        # Return 503 Service Unavailable (not a client error)
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Analysis failed: {str(e)}"
        )

    except Exception as e:
        # Unexpected error — return 500 Internal Server Error
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again."
        )