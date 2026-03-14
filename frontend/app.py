# ================================================================
# app.py — Standalone Streamlit App (merged frontend + backend)
# ================================================================
# For hackathon deployment — everything in one file!
# BERT model + GPT-4o called directly from Streamlit
# No FastAPI needed for the demo link
# ================================================================

import streamlit as st
import torch
import time
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file (for local dev)
# On Streamlit Cloud, secrets come from st.secrets
load_dotenv()

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .verdict-fake {
        background-color: #ff4b4b;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-size: 1.5rem;
        font-weight: bold;
        display: inline-block;
    }
    .verdict-real {
        background-color: #00c853;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-size: 1.5rem;
        font-weight: bold;
        display: inline-block;
    }
    .verdict-uncertain {
        background-color: #ff9800;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-size: 1.5rem;
        font-weight: bold;
        display: inline-block;
    }
    .red-flag {
        background-color: #fff3f3;
        border-left: 4px solid #ff4b4b;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
    }
    .verify-item {
        background-color: #f0f7ff;
        border-left: 4px solid #1e88e5;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Config ───────────────────────────────────────────────────────
# Model name — change this one line to swap models!
BERT_MODEL_NAME = "hamzab/roberta-fake-news-classification"
CONFIDENCE_THRESHOLD = 0.6

# Get OpenAI key from Streamlit secrets (cloud) or .env (local)
def get_openai_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return os.getenv("OPENAI_API_KEY", "")


# ── BERT Model (cached so it loads only once) ────────────────────
@st.cache_resource
def load_bert_model():
    """
    Load BERT model once and cache it.
    @st.cache_resource means this runs only on first call,
    then reuses the same model for all subsequent calls.
    """
    st.info("Loading AI model for the first time... (this takes ~1 min)")

    # Detect best available device
    if torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # Nvidia GPU
    else:
        device = torch.device("cpu")   # CPU fallback

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME)
    model.to(device)
    model.eval()

    return tokenizer, model, device


# ── Stage 1: BERT Detection ──────────────────────────────────────
def run_bert(text: str):
    """
    Run BERT model on the text.
    Returns verdict, confidence, fake_prob, real_prob
    """
    tokenizer, model, device = load_bert_model()

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert to probabilities
    probs = torch.softmax(outputs.logits, dim=1)[0].cpu().tolist()

    # Map to labels
    id2label = model.config.id2label
    label_probs = {id2label[i].upper(): p for i, p in enumerate(probs)}

    # Get fake/real probabilities
    fake_prob = 0.5
    real_prob = 0.5
    for label, prob in label_probs.items():
        if any(x in label for x in ["FAKE", "FALSE", "0"]):
            fake_prob = prob
        elif any(x in label for x in ["REAL", "TRUE", "1"]):
            real_prob = prob

    # Normalize
    total = fake_prob + real_prob
    if total > 0:
        fake_prob /= total
        real_prob /= total

    # Determine verdict
    if fake_prob > real_prob:
        confidence = fake_prob
        verdict = "FAKE" if confidence >= CONFIDENCE_THRESHOLD else "UNCERTAIN"
    else:
        confidence = real_prob
        verdict = "REAL" if confidence >= CONFIDENCE_THRESHOLD else "UNCERTAIN"

    return verdict, round(confidence, 4), round(fake_prob, 4), round(real_prob, 4)


# ── Stage 2: GPT-4o Reasoning ────────────────────────────────────
def run_llm(text: str, verdict: str, confidence: float):
    """
    Call GPT-4o to explain why the article is fake/real/uncertain.
    Returns reasoning, red_flags, what_to_verify
    """
    try:
        client = OpenAI(api_key=get_openai_key())
        confidence_pct = round(confidence * 100, 1)

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            max_tokens=500,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert fact-checker and media literacy educator. "
                        "Analyze news articles for misinformation. "
                        "Always respond with valid JSON only, no extra text."
                    )
                },
                {
                    "role": "user",
                    "content": f"""An AI model classified this news as: {verdict}
Confidence: {confidence_pct}%

NEWS ARTICLE:
\"\"\"{text[:3000]}\"\"\"

Respond with this exact JSON:
{{
    "reasoning": "2-3 sentence explanation of why this is {verdict}",
    "red_flags": [
        "Red flag or manipulative technique #1",
        "Red flag or manipulative technique #2",
        "Red flag or manipulative technique #3"
    ],
    "what_to_verify": [
        "Specific fact-check step #1",
        "Specific fact-check step #2"
    ]
}}"""
                }
            ]
        )

        data = json.loads(response.choices[0].message.content)
        return (
            data.get("reasoning", "No reasoning provided."),
            data.get("red_flags", []),
            data.get("what_to_verify", [])
        )

    except Exception as e:
        return (
            f"AI reasoning unavailable: {str(e)}",
            ["Could not generate red flags"],
            ["Please verify manually on Reuters, AP News, or Snopes"]
        )


# ── UI Rendering Helpers ─────────────────────────────────────────
def render_verdict(verdict, confidence):
    confidence_pct = round(confidence * 100, 1)
    if verdict == "FAKE":
        st.markdown('<span class="verdict-fake">❌ FAKE NEWS</span>', unsafe_allow_html=True)
    elif verdict == "REAL":
        st.markdown('<span class="verdict-real">✅ LIKELY REAL</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="verdict-uncertain">⚠️ UNCERTAIN</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**Model Confidence: {confidence_pct}%**")
    st.progress(confidence)


def render_reasoning(reasoning, red_flags, what_to_verify):
    st.markdown("### 🧠 AI Reasoning")
    st.write(reasoning)

    if red_flags:
        st.markdown("### 🚩 Red Flags Detected")
        for flag in red_flags:
            st.markdown(f'<div class="red-flag">🚩 {flag}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if what_to_verify:
        st.markdown("### 🔎 What To Verify")
        for item in what_to_verify:
            st.markdown(f'<div class="verify-item">🔎 {item}</div>', unsafe_allow_html=True)


# ── Main App ─────────────────────────────────────────────────────
def main():
    # Header
    st.title("🔍 Fake News Detector")
    st.markdown(
        "Paste any news article below. Our AI uses **RoBERTa** for detection "
        "and **GPT-4o** to explain the reasoning."
    )
    st.divider()

    # Example button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Load Example"):
            st.session_state["input_text"] = (
                "BREAKING: Scientists at a secret government lab have "
                "confirmed that 5G towers are being used to control "
                "human thoughts and emotions. Whistleblowers reveal "
                "that major tech companies are complicit in this global "
                "mind control operation. The mainstream media is "
                "suppressing this story. Share before it gets deleted! "
                "Sources close to the deep state confirm this has been "
                "happening since 2019 when the rollout began."
            )

    # Text input
    news_text = st.text_area(
        label="News article text",
        placeholder="Paste the news article text here (minimum 50 characters)...",
        height=200,
        key="input_text",
        label_visibility="collapsed"
    )

    # Character count
    if news_text:
        char_count = len(news_text)
        color = "green" if char_count >= 50 else "red"
        st.markdown(
            f"<small style='color:{color}'>{char_count} characters "
            f"{'✅' if char_count >= 50 else '(minimum 50)'}</small>",
            unsafe_allow_html=True
        )

    # Analyze button
    analyze_clicked = st.button(
        "🔍 Analyze Article",
        type="primary",
        use_container_width=True,
        disabled=len(news_text.strip()) < 50 if news_text else True
    )

    # Run analysis
    if analyze_clicked and news_text:
        start_time = time.time()

        # Stage 1 - BERT
        with st.spinner("🤖 Stage 1: RoBERTa model analyzing..."):
            verdict, confidence, fake_prob, real_prob = run_bert(news_text)

        # Stage 2 - GPT-4o
        with st.spinner("🧠 Stage 2: GPT-4o generating reasoning..."):
            reasoning, red_flags, what_to_verify = run_llm(
                news_text, verdict, confidence
            )

        elapsed = round(time.time() - start_time, 2)

        # Display results
        st.divider()
        st.markdown("## 📊 Analysis Results")
        render_verdict(verdict, confidence)
        st.divider()
        render_reasoning(reasoning, red_flags, what_to_verify)
        st.divider()

        # Technical details for judges
        with st.expander("🔧 Technical Details", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fake Probability", f"{round(fake_prob * 100, 1)}%")
                st.metric("Real Probability", f"{round(real_prob * 100, 1)}%")
            with col2:
                st.metric("Analysis Time", f"{elapsed}s")
                st.metric("LLM", "GPT-4o")
            st.code(BERT_MODEL_NAME, language=None)

    # Footer
    st.markdown(
        '<div class="footer">'
        'Built for hackathon by Team CodeSquad | '
        'RoBERTa + GPT-4o powered | '
        'Always verify from multiple credible sources'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()