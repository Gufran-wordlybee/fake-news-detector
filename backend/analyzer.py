# ================================================================
# analyzer.py — Orchestrates the full 2-stage analysis pipeline
# ================================================================
# This is the brain of the backend. It:
#   1. Takes raw news text
#   2. Sends it to BERT (Stage 1) → gets verdict + confidence
#   3. Sends text + verdict to LLM (Stage 2) → gets reasoning
#   4. Combines everything into one clean AnalyzeResponse
#
# This file never imports OpenAI or BERT directly —
# it only talks to the abstraction layers (bert_detector, get_llm)
# So swapping models never requires touching this file!
# ================================================================

import time
import logging
from backend.models import AnalyzeResponse, Verdict
from backend.ml.bert_detector import detector
from backend.llm import get_llm
from backend.config import BERT_MODEL_NAME, LLM_PROVIDER

logger = logging.getLogger(__name__)


def analyze_news(text: str) -> AnalyzeResponse:
    """
    Main pipeline function — runs the full 2-stage analysis.

    Stage 1: BERT model → verdict + confidence
    Stage 2: LLM → reasoning + red flags (only if FAKE/UNCERTAIN)

    Args:
        text: Raw news article text from the user

    Returns:
        AnalyzeResponse with everything the frontend needs
    """

    # Track total time for the analysis
    start_time = time.time()

    logger.info(f"Starting analysis on text ({len(text)} chars)")

    # ── STAGE 1: BERT Detection ──────────────────────────────────
    logger.info("Stage 1: Running BERT detection...")

    try:
        bert_result = detector.predict(text)
        logger.info(
            f"Stage 1 complete: {bert_result.verdict} "
            f"({bert_result.confidence:.1%} confidence)"
        )
    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")
        raise RuntimeError(f"BERT detection failed: {e}")

    # ── STAGE 2: LLM Reasoning ───────────────────────────────────
    # We ALWAYS call LLM regardless of verdict
    # This gives users explanation for both fake AND real articles
    logger.info(f"Stage 2: Running LLM reasoning with {LLM_PROVIDER}...")

    try:
        llm = get_llm()
        llm_result = llm.generate_reasoning(
            news_text=text,
            verdict=bert_result.verdict.value,   # "FAKE", "REAL", "UNCERTAIN"
            confidence=bert_result.confidence
        )
        logger.info("Stage 2 complete: LLM reasoning generated")

    except Exception as e:
        # LLM failure should NOT crash the whole analysis
        # User still gets the BERT verdict even if LLM is down
        logger.error(f"Stage 2 failed (non-fatal): {e}")
        llm_result = None

    # ── Combine Results ──────────────────────────────────────────
    elapsed_time = round(time.time() - start_time, 2)

    logger.info(f"Analysis complete in {elapsed_time}s")

    return AnalyzeResponse(
        # Final verdict comes from BERT
        verdict=bert_result.verdict,
        confidence=bert_result.confidence,

        # Full stage results
        bert_result=bert_result,
        llm_result=llm_result,

        # Meta info — useful for debugging + showing in UI
        bert_model_used=BERT_MODEL_NAME,
        llm_provider_used=LLM_PROVIDER,
        analysis_time_seconds=elapsed_time
    )