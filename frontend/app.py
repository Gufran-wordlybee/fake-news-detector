# ================================================================
# app.py — Streamlit Frontend for Fake News Detector
# ================================================================
# This is the entire UI — what users see and interact with.
# It:
#   1. Takes news text input from the user
#   2. Sends it to FastAPI backend via HTTP POST
#   3. Displays the verdict, confidence, reasoning, red flags
#
# To run:
#   streamlit run frontend/app.py
# ================================================================

import streamlit as st
import requests
import json
from backend.config import BACKEND_URL

# ── Page Config ──────────────────────────────────────────────────
# Must be the FIRST streamlit command in the file
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",       # cleaner single-column layout
    initial_sidebar_state="collapsed"
)


# ── Custom CSS ───────────────────────────────────────────────────
# Inject some styling to make the UI look polished
st.markdown("""
<style>
    /* Main container padding */
    .main { padding-top: 2rem; }

    /* Verdict badge styling */
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

    /* Red flag item styling */
    .red-flag {
        background-color: #fff3f3;
        border-left: 4px solid #ff4b4b;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
    }

    /* Verify item styling */
    .verify-item {
        background-color: #f0f7ff;
        border-left: 4px solid #1e88e5;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
    }

    /* Footer text */
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────

def check_backend_health() -> bool:
    """
    Ping the FastAPI backend to see if it's running.
    Returns True if healthy, False if unreachable.
    """
    try:
        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=5          # don't wait more than 5 seconds
        )
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def analyze_text(text: str) -> dict:
    """
    Send text to FastAPI /analyze endpoint.
    Returns the full response as a dict, or None on failure.
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}/analyze",
            json={"text": text},
            timeout=60         # LLM can take up to 60s
        )

        if response.status_code == 200:
            return response.json()

        elif response.status_code == 422:
            # Validation error — text too short/long
            detail = response.json().get("detail", [])
            if isinstance(detail, list) and len(detail) > 0:
                st.error(f"Input error: {detail[0].get('msg', 'Invalid input')}")
            return None

        elif response.status_code == 503:
            st.error("Analysis service temporarily unavailable. Please try again.")
            return None

        else:
            st.error(f"Unexpected error (status {response.status_code})")
            return None

    except requests.exceptions.Timeout:
        st.error("Request timed out. The model may be loading — please try again.")
        return None

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Make sure FastAPI server is running.")
        return None


def render_verdict(verdict: str, confidence: float):
    """Renders the big verdict badge + confidence score."""

    confidence_pct = round(confidence * 100, 1)

    if verdict == "FAKE":
        st.markdown(
            '<span class="verdict-fake">❌ FAKE NEWS</span>',
            unsafe_allow_html=True
        )
    elif verdict == "REAL":
        st.markdown(
            '<span class="verdict-real">✅ LIKELY REAL</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="verdict-uncertain">⚠️ UNCERTAIN</span>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Confidence progress bar
    st.markdown(f"**Model Confidence: {confidence_pct}%**")
    st.progress(confidence)


def render_llm_reasoning(llm_result: dict):
    """Renders the LLM reasoning section."""

    if not llm_result:
        st.info("LLM reasoning unavailable for this result.")
        return

    # ── Reasoning paragraph ──────────────────────────────────
    st.markdown("### 🧠 AI Reasoning")
    st.write(llm_result.get("reasoning", "No reasoning provided."))

    # ── Red Flags ────────────────────────────────────────────
    red_flags = llm_result.get("red_flags", [])
    if red_flags:
        st.markdown("### 🚩 Red Flags Detected")
        for flag in red_flags:
            st.markdown(
                f'<div class="red-flag">🚩 {flag}</div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── What to Verify ───────────────────────────────────────
    what_to_verify = llm_result.get("what_to_verify", [])
    if what_to_verify:
        st.markdown("### 🔎 What To Verify")
        for item in what_to_verify:
            st.markdown(
                f'<div class="verify-item">🔎 {item}</div>',
                unsafe_allow_html=True
            )


def render_technical_details(result: dict):
    """
    Shows technical details in an expandable section.
    Great for hackathon judges who want to see under the hood!
    """
    with st.expander("🔧 Technical Details", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Fake Probability",
                f"{round(result['bert_result']['fake_probability'] * 100, 1)}%"
            )
            st.metric(
                "Real Probability",
                f"{round(result['bert_result']['real_probability'] * 100, 1)}%"
            )

        with col2:
            st.metric(
                "Analysis Time",
                f"{result['analysis_time_seconds']}s"
            )
            st.metric(
                "LLM Provider",
                result['llm_provider_used'].upper()
            )

        st.markdown("**BERT Model Used:**")
        st.code(result['bert_model_used'])


# ── Main App ─────────────────────────────────────────────────────

def main():
    # ── Header ───────────────────────────────────────────────
    st.title("🔍 Fake News Detector")
    st.markdown(
        "Paste any news article below. Our AI will analyze it using "
        "a **BERT model** for detection and **GPT-4o** for reasoning."
    )
    st.divider()

    # ── Backend Health Check ─────────────────────────────────
    # Show a warning if backend isn't running
    if not check_backend_health():
        st.warning(
            "⚠️ Backend server is not running. "
            "Please start it with: `uvicorn backend.main:app --reload --port 8000`"
        )
        st.stop()   # Don't render rest of UI if backend is down

    # ── Input Area ───────────────────────────────────────────
    st.markdown("### 📰 Paste News Article")

    # Example button — helps users quickly test the app
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

    # Text input area
    news_text = st.text_area(
        label="News article text",
        placeholder="Paste the news article text here (minimum 50 characters)...",
        height=200,
        key="input_text",
        label_visibility="collapsed"   # hide label, placeholder is enough
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

    # ── Analyze Button ───────────────────────────────────────
    analyze_clicked = st.button(
        "🔍 Analyze Article",
        type="primary",
        use_container_width=True,
        disabled=len(news_text.strip()) < 50 if news_text else True
    )

    # ── Analysis & Results ───────────────────────────────────
    if analyze_clicked and news_text:

        with st.spinner("🔄 Stage 1: BERT model analyzing..."):
            result = analyze_text(news_text)

        if result:
            st.divider()
            st.markdown("## 📊 Analysis Results")

            # Verdict + confidence
            render_verdict(result["verdict"], result["confidence"])
            st.divider()

            # LLM reasoning (shown for all verdicts)
            if result.get("llm_result"):
                render_llm_reasoning(result["llm_result"])
                st.divider()

            # Technical details for judges
            render_technical_details(result)

    # ── Footer ───────────────────────────────────────────────
    st.markdown(
        '<div class="footer">'
        'Built for hackathon | BERT + GPT-4o powered | '
        'Always verify news from multiple credible sources'
        '</div>',
        unsafe_allow_html=True
    )


# Entry point
if __name__ == "__main__":
    main()