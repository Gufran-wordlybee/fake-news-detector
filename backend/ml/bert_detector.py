# ================================================================
# bert_detector.py — Stage 1: ML Model for Fake News Detection
# ================================================================
# This file is responsible for:
#   1. Loading the pretrained BERT/RoBERTa model from HuggingFace
#   2. Taking raw news text as input
#   3. Returning verdict (FAKE/REAL) + confidence score
#
# TO SWAP THE MODEL: just change BERT_MODEL_NAME in config.py
# This file never needs to change — it reads the model name
# dynamically from config!
# ================================================================

import torch
import time
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.config import BERT_MODEL_NAME, CONFIDENCE_THRESHOLD
from backend.models import BertResult, Verdict

# Set up logging so we can see what's happening in the terminal
logger = logging.getLogger(__name__)


class BertDetector:
    """
    Wrapper around any HuggingFace text classification model.
    Designed to be model-agnostic — swap the model in config.py
    and this class just works with the new one automatically.
    """

    def __init__(self):
        # These will be loaded lazily (only when first needed)
        # This avoids slowing down app startup
        self.tokenizer = None
        self.model = None
        self.model_name = BERT_MODEL_NAME
        self.is_loaded = False

        # Detect device: use Apple Silicon GPU if available,
        # otherwise CUDA (Nvidia), otherwise fall back to CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")   # Apple Silicon Mac
            logger.info("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")  # Nvidia GPU
            logger.info("Using CUDA GPU")
        else:
            self.device = torch.device("cpu")   # Fallback
            logger.info("Using CPU")

    def load_model(self):
        """
        Download and load the model from HuggingFace.
        First call downloads it (~500MB), subsequent calls use cache.
        Cache location: ~/.cache/huggingface/
        """
        if self.is_loaded:
            return  # Already loaded, skip

        logger.info(f"Loading model: {self.model_name}")
        logger.info("First run will download the model (~500MB)...")

        try:
            # AutoTokenizer converts raw text → token IDs the model understands
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # AutoModelForSequenceClassification loads a classification model
            # This works for ANY HuggingFace classification model automatically
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )

            # Move model to the best available device
            self.model.to(self.device)

            # Set to evaluation mode — disables dropout layers
            # (dropout is only needed during training, not inference)
            self.model.eval()

            self.is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load BERT model '{self.model_name}': {e}")

    def predict(self, text: str) -> BertResult:
        """
        Main method — takes raw text, returns BertResult.

        Args:
            text: The news article text to analyze

        Returns:
            BertResult with verdict, confidence, and raw probabilities
        """
        # Make sure model is loaded before predicting
        if not self.is_loaded:
            self.load_model()

        try:
            # ── Step 1: Tokenize the input text ─────────────────
            # Converts text → token IDs that the model understands
            # truncation=True: cuts text if longer than model's max length
            # max_length=512: RoBERTa's maximum input size
            # return_tensors="pt": return PyTorch tensors (not numpy)
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # Move input tensors to same device as model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # ── Step 2: Run inference ────────────────────────────
            # torch.no_grad() disables gradient calculation
            # We don't need gradients for inference — saves memory + speed
            with torch.no_grad():
                outputs = self.model(**inputs)

            # ── Step 3: Convert raw scores → probabilities ───────
            # outputs.logits are raw scores (can be any number)
            # softmax converts them to probabilities that sum to 1.0
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

            # Move to CPU and convert to Python list for easy handling
            probs = probabilities[0].cpu().tolist()

            # ── Step 4: Map probabilities to labels ──────────────
            # Get the label mapping from the model's config
            # e.g., {0: "FAKE", 1: "REAL"} or {0: "REAL", 1: "FAKE"}
            # Different models use different orderings — we handle both!
            id2label = self.model.config.id2label

            # Build a clean dict: {"FAKE": 0.87, "REAL": 0.13}
            label_probs = {}
            for idx, prob in enumerate(probs):
                label = id2label[idx].upper()  # normalize to uppercase
                label_probs[label] = prob

            # ── Step 5: Extract fake/real probabilities ──────────
            # Handle models that use different label names
            fake_prob = self._get_prob(label_probs, ["FAKE", "FALSE", "0"])
            real_prob = self._get_prob(label_probs, ["REAL", "TRUE", "1"])

            # Normalize in case labels don't sum to exactly 1.0
            total = fake_prob + real_prob
            if total > 0:
                fake_prob = fake_prob / total
                real_prob = real_prob / total

            # ── Step 6: Determine verdict ────────────────────────
            if fake_prob > real_prob:
                raw_confidence = fake_prob
                if raw_confidence >= CONFIDENCE_THRESHOLD:
                    verdict = Verdict.FAKE
                else:
                    verdict = Verdict.UNCERTAIN
            else:
                raw_confidence = real_prob
                if raw_confidence >= CONFIDENCE_THRESHOLD:
                    verdict = Verdict.REAL
                else:
                    verdict = Verdict.UNCERTAIN

            logger.info(
                f"Prediction: {verdict} "
                f"(fake={fake_prob:.3f}, real={real_prob:.3f})"
            )

            return BertResult(
                verdict=verdict,
                confidence=round(raw_confidence, 4),
                fake_probability=round(fake_prob, 4),
                real_probability=round(real_prob, 4)
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Model prediction failed: {e}")

    def _get_prob(self, label_probs: dict, possible_names: list) -> float:
        """
        Helper to find probability for a label even if model uses
        different naming conventions (FAKE vs FALSE vs 0).

        Args:
            label_probs: dict of label → probability
            possible_names: list of label names to try

        Returns:
            probability value, or 0.5 if none found
        """
        for name in possible_names:
            if name in label_probs:
                return label_probs[name]
        # If no matching label found, return 0.5 (uncertain)
        return 0.5


# ── Singleton Instance ───────────────────────────────────────────
# We create ONE instance and reuse it across all requests
# This avoids reloading the model on every API call (very slow!)
# The model loads once when first request comes in
detector = BertDetector()