# ================================================================
# gemini_llm.py — Google Gemini implementation (ready to swap in)
# ================================================================
# NOT active right now. To activate:
#   1. Set LLM_PROVIDER = "gemini" in config.py
#   2. Add GEMINI_API_KEY to .env
#   3. Uncomment google-generativeai in requirements.txt
#   4. Run: pip install google-generativeai
# That's it — zero other changes needed!
# ================================================================

import logging
import json
from backend.llm.base import BaseLLM
from backend.models import LLMResult
from backend.config import GEMINI_API_KEY, LLM_MODELS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    """Google Gemini implementation — ready to swap in anytime."""

    def __init__(self):
        # Only import if actually used — avoids import error
        # when package isn't installed
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(LLM_MODELS["gemini"])
            logger.info("Gemini LLM initialized")
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Uncomment it in requirements.txt and run pip install."
            )

    def generate_reasoning(self, news_text, verdict, confidence) -> LLMResult:
        # Implementation follows same pattern as OpenAI
        # Fill this in when swapping to Gemini
        raise NotImplementedError("Gemini implementation coming soon")

    def health_check(self) -> bool:
        return False  # Fill in when activating