# ================================================================
# base.py — Abstract Base Class for ALL LLM providers
# ================================================================
# This is the "contract" that every LLM must follow.
# OpenAI, Gemini, Groq — they all implement the same interface.
#
# WHY THIS MATTERS FOR SWAPPING:
# analyzer.py only ever calls base.py methods.
# It doesn't know or care if it's talking to OpenAI or Gemini.
# Swapping = just changing LLM_PROVIDER in config.py.
# ================================================================

from abc import ABC, abstractmethod
from backend.models import LLMResult


class BaseLLM(ABC):
    """
    Abstract base class — every LLM provider must implement this.
    Think of it as a plug shape — all providers fit the same plug.
    """

    @abstractmethod
    def generate_reasoning(
        self,
        news_text: str,
        verdict: str,
        confidence: float
    ) -> LLMResult:
        """
        Given a news article + BERT verdict, generate human-readable
        reasoning explaining WHY it's fake/real/uncertain.

        Args:
            news_text:  The original article text
            verdict:    "FAKE", "REAL", or "UNCERTAIN"
            confidence: BERT's confidence score (0.0 to 1.0)

        Returns:
            LLMResult with reasoning, red_flags, what_to_verify
        """
        pass  # Each provider implements this their own way

    @abstractmethod
    def health_check(self) -> bool:
        """
        Quick check to verify the LLM API is reachable.
        Returns True if working, False if not.
        Used on app startup to catch missing API keys early.
        """
        pass