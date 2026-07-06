"""
ui/streamlit_app.py
-------------------
CallSense-AI — Phase 3.4 UI.
Chains CallIntakeAgent → TranscriptionAgent → SummarizationAgent → QualityScoreAgent.

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
from agents.quality_score_agent import QualityScoreAgent
from agents.summarization_agent import SummarizationAgent
from agents.transcription_agent import TranscriptionAgent
from models.schemas import ProcessingStatus

st.set_page_config(
    page_title="CallSense AI",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Page background ── */
    .stApp { background-color: #0f1117; }

    /* ── Hide default Streamlit header decoration ── */
    header[data-testid="stHeader"] { background: transparent; }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 40px 48px;
        margin-bottom: 32px;
        border: 1px solid #1e3a5f;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-size: 1.1rem;
        color: #8b9dc3;
        margin: 0;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #e2e8f0;
        padding: 12px 0 8px 0;
        border-bottom: 2px solid #1e3a5f;
        margin-bottom: 20px;
        letter-spacing: 0.3px;
    }

    /* ── Generic card ── */
    .card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }

    /* ── Metric card ── */
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        color: #8b9dc3;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .metric-card .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    /* ── Score card ── */
    .score-card {
        background: #1a1f2e;
        border-radius: 12px;
        padding: 18px 20px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 12px;
    }
    .score-card.excellent { border-left-color: #10b981; }
    .score-card.good      { border-left-color: #3b82f6; }
    .score-card.average   { border-left-color: #f59e0b; }
    .score-card.poor      { border-left-color: #ef4444; }
    .score-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .score-value {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .score-value.excellent { color: #10b981; }
    .score-value.good      { color: #3b82f6; }
    .score-value.average   { color: #f59e0b; }
    .score-value.poor      { color: #ef4444; }
    .score-reason {
        font-size: 0.82rem;
        color: #8b9dc3;
        line-height: 1.5;
    }

    /* ── Overall score banner ── */
    .overall-banner {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        margin-bottom: 24px;
    }
    .overall-banner .label {
        font-size: 0.9rem;
        color: #8b9dc3;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .overall-banner .score {
        font-size: 4rem;
        font-weight: 900;
        color: #10b981;
        line-height: 1;
    }
    .overall-banner .out-of {
        font-size: 1.4rem;
        color: #8b9dc3;
        font-weight: 400;
    }
    .overall-banner .grade {
        font-size: 1rem;
        color: #8b9dc3;
        margin-top: 8px;
    }

    /* ── Tag / badge ── */
    .tag {
        display: inline-block;
        background: #1e3a5f;
        color: #60a5fa;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 3px 3px 3px 0;
    }

    /* ── Sentiment badge ── */
    .badge {
        display: inline-block;
        border-radius: 20px;
        padding: 5px 16px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .badge-positive  { background: #064e3b; color: #34d399; }
    .badge-neutral   { background: #451a03; color: #fbbf24; }
    .badge-negative  { background: #450a0a; color: #f87171; }
    .badge-frustrated{ background: #450a0a; color: #f87171; }

    /* ── Bullet list ── */
    .bullet-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 8px 0;
        border-bottom: 1px solid #2d3748;
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    .bullet-item:last-child { border-bottom: none; }
    .bullet-dot { color: #3b82f6; font-size: 1.1rem; margin-top: 1px; }
    .bullet-dot-green { color: #10b981; }
    .bullet-dot-orange { color: #f59e0b; }

    /* ── Pipeline step ── */
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 0.88rem;
        color: #cbd5e1;
    }
    .pipeline-step.done { background: #064e3b22; color: #34d399; }
    .pipeline-step.soon { background: #1a1f2e; color: #4b5563; }

    /* ── Upload zone ── */
    .upload-zone {
        background: #1a1f2e;
        border: 2px dashed #2d3748;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin-bottom: 24px;
    }

    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid #2d3748;
        margin: 28px 0;
    }

    /* ── Streamlit overrides ── */
    div[data-testid="stMetric"] {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="stMetricValue"] { color: #e2e8f0 !important; }
    div[data-testid="stMetricLabel"] { color: #8b9dc3 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }
    .stExpander {
        background: #1a1f2e !important;
        border: 1px solid #2d3748 !important;
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _score_class(score: float) -> str:
    if score >= 9:
        return "excellent"
    if score >= 7:
        return "good"
    if score >= 5:
        return "average"
    return "poor"


def _score_grade(score: float) -> str:
    if score >= 9:
        return "⭐ Excellent"
    if score >= 7:
        return "👍 Good"
    if score >= 5:
        return "📈 Average"
    return "⚠️ Needs Improvement"


def _render_score_card(label: str, score: float, reason: str) -> None:
    cls = _score_class(score)
    st.markdown(
        f"""
        <div class="score-card {cls}">
            <div class="score-title">{label}</div>
            <div class="score-value {cls}">{score:.0f} <span style="font-size:1rem;font-weight:400;color:#8b9dc3;">/ 10</span></div>
            <div class="score-reason">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _progress_colour(score: float) -> str:
    if score >= 9:
        return "#10b981"
    if score >= 7:
        return "#3b82f6"
    if score >= 5:
        return "#f59e0b"
    return "#ef4444"


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:16px 0 8px 0;'>"
        "<span style='font-size:2.5rem;'>📞</span>"
        "<h2 style='color:#e2e8f0;margin:8px 0 0 0;font-size:1.4rem;'>CallSense AI</h2>"
        "<p style='color:#8b9dc3;font-size:0.8rem;margin:4px 0 0 0;'>Call Centre Intelligence Platform</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#2d3748;margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown("<p style='color:#8b9dc3;font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;'>Pipeline Status</p>", unsafe_allow_html=True)
    for step, done in [
        ("🎙️ Intake & Validation", True),
        ("📝 Transcription (Whisper)", True),
        ("🧠 Summarisation (GPT-4o)", True),
        ("📊 Quality Scoring (GPT-4o)", True),
        ("🔀 Routing Decision (GPT-4o)", False),
    ]:
        css = "done" if done else "soon"
        badge = "✅" if done else "🔜"
        st.markdown(
            f"<div class='pipeline-step {css}'>{badge} {step}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#2d3748;margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b9dc3;font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;'>Tech Stack</p>",
        unsafe_allow_html=True,
    )
    for tech in ["OpenAI GPT-4o", "Whisper STT", "LangChain", "LangGraph", "Pydantic v2"]:
        st.markdown(f"<span class='tag'>{tech}</span>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2d3748;margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#4b5563;font-size:0.75rem;text-align:center;'>Version 1.0 · Phase 3.4</p>",
        unsafe_allow_html=True,
    )

# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero">
        <h1>📞 CallSense AI</h1>
        <p>AI-powered call centre assistant — transcribe, summarise, score and route customer support calls automatically.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Upload ─────────────────────────────────────────────────────────────────────

st.markdown("<div class='section-header'>🎙️ Upload Call Recording</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your audio file here",
    type=["mp3", "wav", "m4a", "flac", "mp4"],
    help="Supported: MP3 · WAV · M4A · FLAC · MP4",
    label_visibility="collapsed",
)

if uploaded_file:
    st.markdown(
        f"<div class='card' style='border-left:4px solid #10b981;'>"
        f"<span style='color:#10b981;font-weight:700;'>✅ Ready to process</span>"
        f"<span style='color:#8b9dc3;margin-left:16px;'>{uploaded_file.name}</span>"
        f"<span style='color:#4b5563;margin-left:12px;font-size:0.85rem;'>({uploaded_file.size / 1024:.1f} KB)</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col_btn, col_gap = st.columns([1, 3])
    with col_btn:
        process = st.button("🚀 Process Call", type="primary")

    if process:

        save_dir = Path(tempfile.mkdtemp())
        save_path = save_dir / uploaded_file.name
        save_path.write_bytes(uploaded_file.read())
        state = {"audio_path": str(save_path), "logs": []}

        # Progress bar
        progress = st.progress(0, text="Starting pipeline...")

        with st.spinner("Step 1/4 — Validating audio file..."):
            state.update(CallIntakeAgent().execute(state))
        progress.progress(25, text="✅ Step 1/4 — Intake complete")

        if state.get("status") == ProcessingStatus.FAILED:
            st.error(f"❌ Intake failed: {state.get('error_message')}")
            st.stop()

        with st.spinner("Step 2/4 — Transcribing with Whisper..."):
            state.update(TranscriptionAgent().execute(state))
        progress.progress(50, text="✅ Step 2/4 — Transcription complete")

        if state.get("status") == ProcessingStatus.FAILED:
            st.error(f"❌ Transcription failed: {state.get('error_message')}")
            st.stop()

        with st.spinner("Step 3/4 — Generating summary with GPT-4o..."):
            state.update(SummarizationAgent().execute(state))
        progress.progress(75, text="✅ Step 3/4 — Summary complete")

        if state.get("status") == ProcessingStatus.FAILED:
            st.error(f"❌ Summarisation failed: {state.get('error_message')}")
            st.stop()

        with st.spinner("Step 4/4 — Evaluating call quality with GPT-4o..."):
            state.update(QualityScoreAgent().execute(state))
        progress.progress(100, text="✅ All steps complete!")

        if state.get("status") == ProcessingStatus.FAILED:
            st.error(f"❌ Quality scoring failed: {state.get('error_message')}")
            st.stop()

        st.markdown(
            "<div class='card' style='border-left:4px solid #10b981;text-align:center;'>"
            "<span style='color:#10b981;font-size:1.1rem;font-weight:700;'>🎉 Processing Complete — All 4 agents ran successfully</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Tabs ──────────────────────────────────────────────────────────────

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📋 Metadata & Transcript", "🧠 Call Summary", "📊 Quality Score Card", "🪵 Agent Logs"]
        )

        # ── Tab 1: Metadata & Transcript ──────────────────────────────────────

        with tab1:
            st.markdown("<div class='section-header'>📋 File Metadata</div>", unsafe_allow_html=True)
            m = state["metadata"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("📁 File", m.file_name)
            c2.metric("🎵 Format", m.file_type.upper())
            c3.metric("💾 Size", f"{m.file_size_bytes / 1024:.1f} KB")
            c4.metric("🌐 Language", (m.language or "en").upper())
            c5.metric("📝 Words", len(state.get("transcript", "").split()))

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>📝 Transcript</div>", unsafe_allow_html=True)

            transcript = state.get("transcript", "")
            st.text_area(
                label="transcript",
                value=transcript,
                height=320,
                label_visibility="collapsed",
            )
            st.download_button(
                label="⬇️ Download Transcript (.txt)",
                data=transcript,
                file_name=f"{Path(uploaded_file.name).stem}_transcript.txt",
                mime="text/plain",
            )

        # ── Tab 2: Summary ────────────────────────────────────────────────────

        with tab2:
            s = state["summary"]
            sentiment_key = s.customer_sentiment.lower()
            sentiment_badge_class = f"badge-{sentiment_key}" if sentiment_key in ("positive", "neutral", "negative", "frustrated") else "badge-neutral"
            sentiment_icon = {"positive": "😊", "neutral": "😐", "negative": "😞", "frustrated": "😤"}.get(sentiment_key, "😐")

            st.markdown("<div class='section-header'>🧠 Call Summary</div>", unsafe_allow_html=True)

            st.markdown(
                f"<div class='card'>"
                f"<p style='color:#e2e8f0;font-size:1rem;line-height:1.7;margin:0 0 12px 0;'>{s.summary}</p>"
                f"<span class='badge {sentiment_badge_class}'>{sentiment_icon} {s.customer_sentiment.capitalize()}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"<div class='card' style='border-left:4px solid #3b82f6;'>"
                    f"<div style='color:#8b9dc3;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;'>🔍 Customer Issue</div>"
                    f"<div style='color:#e2e8f0;font-size:0.95rem;line-height:1.6;'>{s.customer_issue}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='card' style='border-left:4px solid #10b981;'>"
                    f"<div style='color:#8b9dc3;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;'>✅ Resolution</div>"
                    f"<div style='color:#e2e8f0;font-size:0.95rem;line-height:1.6;'>{s.resolution}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col2:
                action_html = "".join(
                    f"<div class='bullet-item'><span class='bullet-dot bullet-dot-orange'>›</span><span>{item}</span></div>"
                    for item in s.action_items
                ) or "<span style='color:#4b5563;font-size:0.85rem;'>No action items identified.</span>"

                st.markdown(
                    f"<div class='card'>"
                    f"<div style='color:#8b9dc3;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:12px;'>📋 Action Items</div>"
                    f"{action_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                topics_html = " ".join(f"<span class='tag'>{t}</span>" for t in s.key_topics) or "<span style='color:#4b5563;'>None identified.</span>"
                st.markdown(
                    f"<div class='card'>"
                    f"<div style='color:#8b9dc3;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:12px;'>🏷️ Key Topics</div>"
                    f"{topics_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── Tab 3: Quality Score Card ─────────────────────────────────────────

        with tab3:
            qs = state["quality_score"]

            # Overall score banner
            grade = _score_grade(qs.overall_score)
            st.markdown(
                f"<div class='overall-banner'>"
                f"<div class='label'>Overall Quality Score</div>"
                f"<div class='score'>{qs.overall_score:.1f}<span class='out-of'> / 10</span></div>"
                f"<div class='grade'>{grade}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # 6 dimension score cards in 2 columns
            st.markdown("<div class='section-header'>📝 Dimension Breakdown</div>", unsafe_allow_html=True)

            dimensions = [
                ("🤝 Empathy", qs.empathy_score, qs.empathy_reason),
                ("👔 Professionalism", qs.professionalism_score, qs.professionalism_reason),
                ("💬 Communication Clarity", qs.communication_clarity_score, qs.communication_clarity_reason),
                ("🔍 Problem Understanding", qs.problem_understanding_score, qs.problem_understanding_reason),
                ("✅ Resolution Quality", qs.resolution_quality_score, qs.resolution_quality_reason),
                ("📋 Compliance", qs.compliance_score, qs.compliance_reason),
            ]

            left_col, right_col = st.columns(2)
            for i, (label, score, reason) in enumerate(dimensions):
                with left_col if i % 2 == 0 else right_col:
                    _render_score_card(label, score, reason)
                    pct = int(score * 10)
                    colour = _progress_colour(score)
                    st.markdown(
                        f"<div style='background:#2d3748;border-radius:4px;height:6px;margin:-8px 0 16px 0;'>"
                        f"<div style='background:{colour};width:{pct}%;height:6px;border-radius:4px;'></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # Strengths & Improvements
            str_col, imp_col = st.columns(2)

            with str_col:
                strengths_html = "".join(
                    f"<div class='bullet-item'><span class='bullet-dot bullet-dot-green'>✓</span><span>{s}</span></div>"
                    for s in qs.strengths
                )
                st.markdown(
                    f"<div class='card' style='border-left:4px solid #10b981;'>"
                    f"<div style='color:#10b981;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:12px;font-weight:700;'>💪 Strengths</div>"
                    f"{strengths_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with imp_col:
                improvements_html = "".join(
                    f"<div class='bullet-item'><span class='bullet-dot' style='color:#f59e0b;'>→</span><span>{a}</span></div>"
                    for a in qs.improvement_areas
                )
                st.markdown(
                    f"<div class='card' style='border-left:4px solid #f59e0b;'>"
                    f"<div style='color:#f59e0b;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:12px;font-weight:700;'>🎯 Areas for Improvement</div>"
                    f"{improvements_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # Overall feedback
            st.markdown(
                f"<div class='card' style='border-left:4px solid #8b5cf6;'>"
                f"<div style='color:#8b5cf6;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:10px;font-weight:700;'>💼 Overall Feedback</div>"
                f"<div style='color:#cbd5e1;font-size:0.95rem;line-height:1.8;'>{qs.overall_feedback}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Tab 4: Agent Logs ─────────────────────────────────────────────────

        with tab4:
            st.markdown("<div class='section-header'>🪵 Agent Logs</div>", unsafe_allow_html=True)
            for entry in state.get("logs", []):
                st.code(entry, language="text")

else:
    st.markdown(
        "<div class='card' style='text-align:center;padding:48px 32px;border:2px dashed #2d3748;'>"
        "<div style='font-size:3rem;margin-bottom:16px;'>🎙️</div>"
        "<div style='color:#e2e8f0;font-size:1.1rem;font-weight:600;margin-bottom:8px;'>Upload a call recording to get started</div>"
        "<div style='color:#4b5563;font-size:0.88rem;'>Supported formats: MP3 · WAV · M4A · FLAC · MP4</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown(
    "<hr style='border-color:#2d3748;margin-top:40px;'>"
    "<p style='text-align:center;color:#4b5563;font-size:0.78rem;padding:8px 0 16px 0;'>"
    "CallSense AI · Version 1.0 · Phase 3.4 · Powered by OpenAI GPT-4o &amp; Whisper"
    "</p>",
    unsafe_allow_html=True,
)
