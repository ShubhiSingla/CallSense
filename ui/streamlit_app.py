"""
ui/streamlit_app.py
-------------------
CallSense-AI — Phase 3.3 UI.
Chains CallIntakeAgent → TranscriptionAgent → SummarizationAgent.

Run with:
    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from agents.intake_agent import CallIntakeAgent
from agents.summarization_agent import SummarizationAgent
from agents.transcription_agent import TranscriptionAgent
from models.schemas import ProcessingStatus

st.set_page_config(
    page_title="CallSense AI",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📞 CallSense AI")
    st.markdown("---")
    st.markdown("### 📖 Project Overview")
    st.markdown(
        "CallSense AI is an **AI-powered call centre assistant** that "
        "automatically transcribes, summarises, scores, and routes "
        "customer support calls using Whisper and GPT-4o."
    )
    st.markdown("---")
    st.markdown(
        "**Pipeline**\n"
        "1. 🎙️ Intake & Validation ✅\n"
        "2. 📝 Transcription (Whisper) ✅\n"
        "3. 🧠 Summarisation (GPT-4o) ✅\n"
        "4. 📊 Quality Scoring (GPT-4o) — coming soon\n"
        "5. 🔀 Routing Decision (GPT-4o) — coming soon"
    )
    st.markdown("---")
    st.caption("Version 1.0  ·  Phase 3.3")

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("📞 CallSense AI")
st.subheader("AI Powered Call Center Assistant")
st.markdown("---")

# ── Audio Upload ───────────────────────────────────────────────────────────────

st.markdown("### 🎙️ Upload Call Recording")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["mp3", "wav", "m4a", "flac", "mp4"],
    help="Supported formats: MP3, WAV, M4A, FLAC, MP4",
)

if uploaded_file:
    st.success(f"✅ File received: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

    if st.button("🚀 Process Call", type="primary"):

        # Save uploaded file to disk using the original filename
        save_dir = Path(tempfile.mkdtemp())
        save_path = save_dir / uploaded_file.name
        save_path.write_bytes(uploaded_file.read())

        state = {"audio_path": str(save_path), "logs": []}

        # ── Step 1: Intake Agent ───────────────────────────────────────────────

        with st.spinner("Step 1/3 — Validating audio file..."):
            state.update(CallIntakeAgent().execute(state))

        if state.get("status") == ProcessingStatus.FAILED:
            st.error(f"❌ Intake failed: {state.get('error_message')}")
            st.stop()

        # ── Step 2: Transcription Agent ────────────────────────────────────────

        with st.spinner("Step 2/3 — Transcribing audio with Whisper..."):
            state.update(TranscriptionAgent().execute(state))

        if state.get("status") == ProcessingStatus.FAILED:
            st.error(f"❌ Transcription failed: {state.get('error_message')}")
            st.stop()

        # ── Step 3: Summarization Agent ────────────────────────────────────────

        with st.spinner("Step 3/3 — Generating summary with GPT-4o..."):
            state.update(SummarizationAgent().execute(state))

        if state.get("status") == ProcessingStatus.FAILED:
            st.error(f"❌ Summarisation failed: {state.get('error_message')}")
            st.stop()

        st.success("✅ Processing complete!")
        st.markdown("---")

        # ── File Metadata ──────────────────────────────────────────────────────

        st.markdown("### 📋 File Metadata")
        m = state["metadata"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("File Name", m.file_name)
        c2.metric("File Type", m.file_type.upper())
        c3.metric("File Size", f"{m.file_size_bytes / 1024:.1f} KB")
        c4.metric("Language", m.language or "en")
        c5.metric("Word Count", len(state.get("transcript", "").split()))

        st.markdown("---")

        # ── Transcript ─────────────────────────────────────────────────────────

        st.markdown("### 📝 Transcript")
        transcript = state.get("transcript", "")
        with st.expander("View full transcript", expanded=False):
            st.text_area(label="transcript", value=transcript, height=250, label_visibility="collapsed")
            st.download_button(
                label="⬇️ Download Transcript",
                data=transcript,
                file_name=f"{Path(uploaded_file.name).stem}_transcript.txt",
                mime="text/plain",
            )

        st.markdown("---")

        # ── Summary ────────────────────────────────────────────────────────────

        st.markdown("### 🧠 Call Summary")
        s = state["summary"]

        # Sentiment badge colour
        sentiment_colour = {
            "positive": "🟢", "neutral": "🟡",
            "negative": "🔴", "frustrated": "🔴",
        }.get(s.customer_sentiment.lower(), "⚪")

        st.markdown(f"**{s.summary}**")
        st.markdown(f"**Sentiment:** {sentiment_colour} {s.customer_sentiment.capitalize()}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔍 Customer Issue")
            st.info(s.customer_issue)

            st.markdown("#### ✅ Resolution")
            st.success(s.resolution)

        with col2:
            st.markdown("#### 📋 Action Items")
            if s.action_items:
                for item in s.action_items:
                    st.markdown(f"- {item}")
            else:
                st.markdown("_No action items identified._")

            st.markdown("#### 🏷️ Key Topics")
            if s.key_topics:
                st.markdown(" ".join([f"`{t}`" for t in s.key_topics]))
            else:
                st.markdown("_No key topics identified._")

        st.markdown("---")

        # ── Quality Score placeholder ──────────────────────────────────────────

        st.markdown("### 📊 Quality Score")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Overall", "—")
        q2.metric("Empathy", "—")
        q3.metric("Resolution", "—")
        q4.metric("Communication", "—")
        st.info("⏳ Coming in Phase 3.4 — Quality Score Agent.")

        st.markdown("---")

        # ── Agent Logs ─────────────────────────────────────────────────────────

        with st.expander("🪵 Agent Logs"):
            for entry in state.get("logs", []):
                st.code(entry)

else:
    st.info("👆 Upload an audio file above to begin.")

# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption("CallSense AI · Version 1.0 · Phase 3.3 · Powered by OpenAI & LangGraph")
