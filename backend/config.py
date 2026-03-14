# ================================================================
# config.py — Central configuration for the entire project
# ================================================================
# THIS IS THE ONLY FILE YOU NEED TO TOUCH TO:
#   1. Swap the ML model  → change BERT_MODEL_NAME
#   2. Swap the LLM       → change LLM_PROVIDER
# Nothing else needs to change anywhere in the codebase!
# ================================================================

import os
from dotenv import load_dotenv

# Load all variables from .env file into environment
load_dotenv()


# ── ML Model Config ────────────────────────────────────────────
# This is the HuggingFace model ID for fake news detection
# To swap model: change ONLY this line, nothing else!
# Other good options to try later:
#   "GonzaloA/fake-news-detection"
#   "jy46604790/Fake-News-Bert-Detect"
BERT_MODEL_NAME = "hamzab/roberta-fake-news-classification"

# Confidence threshold — if model confidence is below this,
# we mark result as "Uncertain" instead of Fake/Real
CONFIDENCE_THRESHOLD = 0.6


# ── LLM Config ─────────────────────────────────────────────────
# To swap LLM provider: change ONLY this line!
# Options: "openai" | "gemini" | "groq"
LLM_PROVIDER = "openai"

# LLM model names for each provider
# When you swap LLM_PROVIDER, the right model is picked automatically
LLM_MODELS = {
    "openai": "gpt-4o",
    "gemini": "gemini-1.5-pro",  # ready for future swap
    "groq":   "llama3-8b-8192",  # ready for future swap
}

# How creative/focused the LLM response should be
# 0.0 = very focused, 1.0 = more creative
# We keep it low (0.2) for factual reasoning
LLM_TEMPERATURE = 0.2

# Maximum words the LLM can use for its reasoning response
LLM_MAX_TOKENS = 500


# ── API Keys ────────────────────────────────────────────────────
# These are read from your .env file — never hardcoded here!
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")


# ── Server Config ───────────────────────────────────────────────
# FastAPI server settings
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000

# This is the URL Streamlit uses to talk to FastAPI
# During local development this stays as localhost
# When deploying, we'll update this to the live URL
BACKEND_URL = os.getenv("BACKEND_URL", f"http://localhost:{BACKEND_PORT}")


# ── Validation ──────────────────────────────────────────────────
# This runs when the app starts — catches missing keys early
# so you get a clear error message instead of a cryptic crash
def validate_config():
    """Check that required config values are present."""

    # Check LLM provider is valid
    if LLM_PROVIDER not in LLM_MODELS:
        raise ValueError(
            f"Invalid LLM_PROVIDER '{LLM_PROVIDER}'. "
            f"Must be one of: {list(LLM_MODELS.keys())}"
        )

    # Check the right API key exists for chosen provider
    api_keys = {
        "openai": OPENAI_API_KEY,
        "gemini": GEMINI_API_KEY,
        "groq":   GROQ_API_KEY,
    }
    if not api_keys[LLM_PROVIDER]:
        raise ValueError(
            f"Missing API key for provider '{LLM_PROVIDER}'. "
            f"Add it to your .env file."
        )