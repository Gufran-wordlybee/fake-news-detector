# ================================================================
# openai_llm.py — OpenAI GPT-4o implementation of BaseLLM
# ================================================================
# This is the CURRENT active LLM provider.
# To swap to Gemini: set LLM_PROVIDER = "gemini" in config.py
# This file will no longer be called — gemini_llm.py takes over.
# ================================================================

import logging
import json
from openai import OpenAI
from backend.llm.base import BaseLLM
from backend.models import LLMResult
from backend.config import OPENAI_API_KEY, LLM_MODELS, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """GPT-4o implementation of the LLM reasoning layer."""

    def __init__(self):
        # Initialize the OpenAI client with our API key
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        # Get the model name from config (e.g., "gpt-4o")
        self.model = LLM_MODELS["openai"]

        logger.info(f"OpenAI LLM initialized with model: {self.model}")

    def generate_reasoning(
        self,
        news_text: str,
        verdict: str,
        confidence: float
    ) -> LLMResult:
        """
        Calls GPT-4o to generate reasoning for the BERT verdict.
        Returns structured LLMResult with reasoning + red flags.
        """
        try:
            # Build the prompt — this is carefully engineered to
            # return consistent, structured JSON every time
            prompt = self._build_prompt(news_text, verdict, confidence)

            logger.info(f"Calling GPT-4o for reasoning (verdict={verdict})")

            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=LLM_TEMPERATURE,  # low = more focused/factual
                max_tokens=LLM_MAX_TOKENS,
                # Tell GPT to always return valid JSON
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": self._system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Extract the text response
            raw_response = response.choices[0].message.content
            logger.info("GPT-4o response received successfully")

            # Parse the JSON response into our LLMResult model
            return self._parse_response(raw_response)

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            # Return a fallback result instead of crashing
            return self._fallback_result(str(e))

    def health_check(self) -> bool:
        """Quick ping to verify OpenAI API is reachable."""
        try:
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False

    def _system_prompt(self) -> str:
        """
        The system prompt that sets GPT's behavior.
        We tell it to be a fact-checker and always return JSON.
        """
        return """You are an expert fact-checker and media literacy educator 
specializing in detecting misinformation and fake news.

Your job is to analyze news articles and explain WHY they may be 
fake, real, or uncertain based on linguistic patterns, logical 
consistency, emotional manipulation, and factual plausibility.

You MUST respond with valid JSON only. No extra text outside the JSON.
Be specific, educational, and helpful to the reader."""

    def _build_prompt(
        self,
        news_text: str,
        verdict: str,
        confidence: float
    ) -> str:
        """
        Builds the user prompt with the article + BERT verdict.
        We tell GPT what BERT found so it focuses its reasoning.
        """
        confidence_pct = round(confidence * 100, 1)

        return f"""An AI model analyzed this news article and classified it as: {verdict}
Confidence: {confidence_pct}%

NEWS ARTICLE:
\"\"\"
{news_text[:3000]}
\"\"\"

Based on this article and the AI verdict, provide your analysis.

Respond with this exact JSON structure:
{{
    "reasoning": "A clear 2-3 sentence explanation of why this article appears to be {verdict}. Be specific about what in the text led to this conclusion.",
    "red_flags": [
        "Specific red flag or manipulative technique #1",
        "Specific red flag or manipulative technique #2",
        "Specific red flag or manipulative technique #3"
    ],
    "what_to_verify": [
        "Specific thing to fact-check #1",
        "Specific thing to fact-check #2"
    ]
}}

For red_flags: identify specific techniques like emotional manipulation, 
missing sources, vague attribution, sensational language, logical fallacies etc.
For what_to_verify: suggest concrete steps like checking specific claims,
looking up named sources, verifying statistics etc.
If verdict is REAL, still provide analysis but focus on credibility indicators."""

    def _parse_response(self, raw_response: str) -> LLMResult:
        """
        Parses GPT's JSON response into our LLMResult model.
        Has fallback handling if JSON is malformed.
        """
        try:
            data = json.loads(raw_response)

            return LLMResult(
                reasoning=data.get("reasoning", "No reasoning provided."),
                red_flags=data.get("red_flags", []),
                what_to_verify=data.get("what_to_verify", [])
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT response as JSON: {e}")
            # If JSON parsing fails, return the raw text as reasoning
            return LLMResult(
                reasoning=raw_response,
                red_flags=["Could not parse structured response"],
                what_to_verify=["Please verify this article manually"]
            )

    def _fallback_result(self, error_msg: str) -> LLMResult:
        """
        Returns a safe fallback when API call completely fails.
        Better than crashing the whole app over LLM failure.
        """
        return LLMResult(
            reasoning=(
                "The AI reasoning service is temporarily unavailable. "
                "Please rely on the confidence score from the detection model."
            ),
            red_flags=["LLM reasoning unavailable"],
            what_to_verify=[
                "Cross-check with Reuters, AP News, or Snopes",
                "Search for the same story on multiple credible outlets"
            ]
        )