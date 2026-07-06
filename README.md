<div align="center">

# CallSense AI

**A Multi-Agent, Multimodal AI Assistant for Call Center Analytics**

Turns raw customer support call recordings into structured business insights — transcripts, summaries, sentiment, and quality scorecards — in seconds.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green)
![LangChain](https://img.shields.io/badge/LangChain-LLM-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-purple)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-red)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

## Overview

Call centers generate thousands of recorded conversations every day, but reviewing them manually — listening to audio, writing summaries, scoring agent performance — doesn't scale.

**CallSense AI** automates that entire review process. Upload a call recording, and a pipeline of specialized AI agents transcribes it, summarizes it, and evaluates the support representative's performance, so supervisors can spend their time acting on insights instead of producing them.

The system is built as a **modular multi-agent architecture**: each agent owns one responsibility, communicates through shared state, and can be extended or replaced independently.

---

## Architecture

```text
                    Upload Audio
                         │
                         ▼
                 ┌──────────────────┐
                 │   Intake Agent    │  validates file, extracts metadata
                 └────────┬──────────┘
                          │
              ┌───────────┴───────────┐
            Valid                  Invalid
              │                       │
              ▼                       ▼
     ┌──────────────────┐      Return Error
     │  Transcription    │  Whisper: speech → text
     └────────┬──────────┘
              ▼
     ┌──────────────────┐
     │  Summarization    │  GPT-4o: issue, resolution, sentiment, topics
     └────────┬──────────┘
              ▼
     ┌──────────────────┐
     │  Quality Score    │  GPT-4o: agent performance scorecard
     └────────┬──────────┘
              ▼
     ┌──────────────────┐
     │     Routing       │  (upcoming) approve / escalate for review
     └────────┬──────────┘
              ▼
     ┌──────────────────┐
     │   Streamlit UI    │  transcript · summary · scorecard · routing
     └──────────────────┘
```

---

## Features

### Implemented

| Agent | Responsibility |
|---|---|
| **Intake** | Validates uploaded audio, supports multiple formats, extracts file metadata, handles bad uploads gracefully |
| **Transcription** | Converts speech to text via OpenAI Whisper, detects language, stores transcript in shared state |
| **Summarization** | Uses GPT-4o + LangChain/Pydantic structured output to extract the customer issue, resolution, sentiment, action items, and key topics |
| **Quality Score** | Scores the representative on empathy, professionalism, clarity, problem understanding, resolution quality, and compliance, with strengths, gaps, and detailed QA feedback |

**Shared foundation:** LangGraph state management, Pydantic data validation, a modular service layer, centralized logging, and robust error handling throughout.

### Upcoming

Routing agent · full LangGraph orchestration · interactive Streamlit dashboard · Docker deployment · LangSmith tracing · speaker diarization · multi-language support · intent classification · PDF report export · call search & filtering.

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| AI Framework | LangChain |
| Multi-Agent Framework | LangGraph |
| LLM | OpenAI GPT-4o |
| Speech-to-Text | OpenAI Whisper |
| Data Validation | Pydantic |
| Frontend | Streamlit |
| Config | YAML + environment variables |
| Testing | PyTest |
| Logging | Python `logging` |

---

## Project Structure

```text
CallSense-AI/
├── agents/
│   ├── intake_agent.py
│   ├── transcription_agent.py
│   ├── summarization_agent.py
│   ├── quality_score_agent.py
│   └── routing_agent.py
├── config/
│   └── settings.py
├── data/
│   ├── audio/
│   ├── transcripts/
│   └── outputs/
├── graph/
│   ├── state.py
│   └── workflow.py
├── models/
│   └── schemas.py
├── prompts/
│   ├── summary_prompt.py
│   └── qa_prompt.py
├── services/
│   ├── openai_service.py
│   └── whisper_service.py
├── ui/
│   └── streamlit_app.py
├── tests/
├── utils/
├── main.py
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-github-username>/CallSense-AI.git
cd CallSense-AI

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
```

Then edit `.env`:

```text
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=CallSenseAI
LANGCHAIN_TRACING_V2=false
```

### Verify & Run

```bash
# Confirm your OpenAI connection
python test_openai.py
# Expected: [SUCCESS] OpenAI Connected Successfully

# Launch the app
streamlit run ui/streamlit_app.py
```

Then open **http://localhost:8501**.

---

## Example

**Call transcript**

> **Customer:** Hello, I haven't received my refund.
> **Agent:** I'm sorry about the delay. I've initiated your refund and it will be credited within three business days.
> **Customer:** Thank you.

**AI-generated summary**

| Field | Output |
|---|---|
| **Summary** | Customer reported a delayed refund; agent apologized, confirmed the refund was initiated, and gave a three-business-day timeline. |
| **Issue** | Refund not received |
| **Resolution** | Refund initiated; expected within 3 business days |
| **Action Items** | Wait up to 3 business days; follow up with support if not received |
| **Sentiment** | 🟡 Neutral |
| **Key Topics** | Refund, Payment, Customer Support |

**Quality scorecard**

| Metric | Score |
|---|:---:|
| Empathy | 9/10 |
| Professionalism | 10/10 |
| Communication Clarity | 9/10 |
| Problem Understanding | 10/10 |
| Resolution Quality | 9/10 |
| Compliance | 9/10 |
| **Overall Score** | **9.3/10** |

- **Strengths:** strong empathy, professional tone, clear timeline, effective resolution.
- **Areas for improvement:** offer proactive status updates; share a reference number for follow-up.
- **Feedback:** the representative handled the interaction professionally and empathetically, correctly identified the issue, and communicated a realistic resolution timeline — meeting expected service standards.

---

## Roadmap

| Phase | Focus |
|---|---|
| **4** | Connect all agents via LangGraph, state-based orchestration, conditional routing, workflow visualization |
| **5** | Interactive Streamlit dashboard — audio upload, transcript viewer, summary & QA dashboards, routing decisions |
| **6** | Docker deployment, LangSmith tracing, end-to-end testing, CI/CD, performance monitoring |
| **Beyond** | Speaker diarization, multi-language support, intent classification, supervisor analytics, call search, PDF export, email notifications, human review queue |

**Current status:** setup, OpenAI integration, Pydantic models, LangGraph state, and the Intake/Transcription/Summarization/Quality Score agents are complete. Routing, full workflow orchestration, the dashboard, and deployment are in progress.

---

## Key Learnings

Building CallSense AI involved hands-on work with:

- Designing modular multi-agent AI systems and state-driven workflows with LangGraph
- Integrating OpenAI Whisper for speech-to-text and GPT-4o for structured summarization and evaluation
- Building reliable data contracts with Pydantic and structured outputs with LangChain
- Designing production-ready service layers with clean logging, validation, and error handling

---

## Contributing

Contributions are welcome — bug reports, feature suggestions, documentation improvements, and pull requests all help.

## License

Licensed under the [MIT License](LICENSE).

## Author

**Shubhi Singla**

---

<div align="center">

⭐ If this project helped you, consider giving it a star on GitHub!

</div>
