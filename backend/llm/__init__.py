# ================================================================
# __init__.py — LLM Factory
# ================================================================
# This is where the swap magic actually happens at runtime.
# Reads LLM_PROVIDER from config and returns the right class.
# analyzer.py calls get_llm() and never knows which provider it got.
# ================================================================

import logging
from backend.config import LLM_PROVIDER
from backend.llm.base import BaseLLM

logger = logging.getLogger(__name__)


def get_llm() -> BaseLLM:
    """
    Factory function — returns the correct LLM based on config.
    This is the ONLY place that knows about specific providers.

    Returns:
        An instance of the configured LLM provider

    Raises:
        ValueError if LLM_PROVIDER is not recognized
    """
    logger.info(f"Loading LLM provider: {LLM_PROVIDER}")

    if LLM_PROVIDER == "openai":
        from backend.llm.openai_llm import OpenAILLM
        return OpenAILLM()

    elif LLM_PROVIDER == "gemini":
        from backend.llm.gemini_llm import GeminiLLM
        return GeminiLLM()

    elif LLM_PROVIDER == "groq":
        from backend.llm.groq_llm import GroqLLM
        return GroqLLM()

    else:
        raise ValueError(
            f"Unknown LLM provider: '{LLM_PROVIDER}'. "
            f"Must be one of: openai, gemini, groq"
        )