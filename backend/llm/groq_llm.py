# ================================================================
# groq_llm.py — Groq implementation (ready to swap in)
# ================================================================
# NOT active right now. To activate:
#   1. Set LLM_PROVIDER = "groq" in config.py
#   2. Add GROQ_API_KEY to .env
#   3. Uncomment groq in requirements.txt
#   4. Run: pip install groq
# That's it — zero other changes needed!
# ================================================================

import logging
from backend.llm.base import BaseLLM
from backend.models import LLMResult
from backend.config import GROQ_API_KEY, LLM_MODELS

logger = logging.getLogger(__name__)


class GroqLLM(BaseLLM):
    """Groq LLaMA implementation — ready to swap in anytime."""

    def __init__(self):
        try:
            from groq import Groq
            self.client = Groq(api_key=GROQ_API_KEY)
            self.model = LLM_MODELS["groq"]
            logger.info("Groq LLM initialized")
        except ImportError:
            raise ImportError(
                "groq not installed. "
                "Uncomment it in requirements.txt and run pip install."
            )

    def generate_reasoning(self, news_text, verdict, confidence) -> LLMResult:
        # Implementation follows same pattern as OpenAI
        # Fill this in when swapping to Groq
        raise NotImplementedError("Groq implementation coming soon")

    def health_check(self) -> bool:
        return False  # Fill in when activating