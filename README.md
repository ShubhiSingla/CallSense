# CallSense AI

> AI-powered Call Centre Assistant — transcribe, summarise, score, and route customer support calls automatically.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| pip | latest |
| OpenAI API Key | required |

---

## Python Version

This project requires **Python 3.11**.

Verify your version:

```bash
python --version
```

---

## Virtual Environment

### Create

**macOS / Linux**
```bash
python3.11 -m venv .venv
```

**Windows**
```bash
python -m venv .venv
```

### Activate

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

---

## Installing Dependencies

```bash
pip install -r requirements.txt
```

---

## Creating .env

```bash
cp .env.example .env
```

Then open `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-...
LANGCHAIN_API_KEY=ls__...        # optional — for LangSmith tracing
LANGCHAIN_PROJECT=CallSenseAI
LANGCHAIN_TRACING_V2=false
```

---

## Running the OpenAI Test

Verify your API key and GPT-4o connection:

```bash
python test_openai.py
```

Expected output:
```
[SUCCESS] Hello from CallSense AI! ...
```

---

## Running Streamlit

```bash
streamlit run ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
CallSense-AI/
│
├── agents/                      # Agent implementations (coming soon)
│   ├── intake_agent.py
│   ├── transcription_agent.py
│   ├── summarization_agent.py
│   ├── quality_score_agent.py
│   └── routing_agent.py
│
├── config/
│   └── settings.py              # Env-backed Settings class
│
├── graph/
│   ├── state.py                 # CallSenseState TypedDict
│   └── workflow.py              # LangGraph StateGraph
│
├── models/
│   └── schemas.py               # Pydantic v2 models
│
├── prompts/
│   ├── summary_prompt.py
│   └── qa_prompt.py
│
├── services/
│   ├── openai_service.py        # GPT-4o wrapper
│   └── whisper_service.py       # Whisper wrapper
│
├── utils/
│   ├── logger.py
│   └── validator.py
│
├── ui/
│   └── streamlit_app.py         # Streamlit front-end
│
├── data/
│   ├── audio/
│   ├── transcripts/
│   └── outputs/
│
├── tests/
│   ├── test_intake.py
│   └── test_summary.py
│
├── main.py                      # App entry point
├── test_openai.py               # OpenAI connection test
├── requirements.txt
├── .env.example
└── README.md
```

---

## Phase 3.3 — Summarization Agent

### Purpose

The Summarization Agent is the third node in the pipeline. It reads the transcript from `CallState`, sends it to GPT-4o via LangChain structured output, and writes a validated `CallSummary` Pydantic model back into `CallState`.

### Input

`CallState` containing `transcript` populated by the Transcription Agent.

### Output

On success:

| Field | Value |
|-------|-------|
| `summary` | Validated `CallSummary` Pydantic model |
| `status` | `ProcessingStatus.PROCESSING` |
| `error_message` | `None` |
| `logs` | Log entry appended |

On failure:

| Field | Value |
|-------|-------|
| `status` | `ProcessingStatus.FAILED` |
| `error_message` | Human-readable description |
| `logs` | Error entry appended |

### Prompt Design

The system prompt instructs GPT-4o to act as a call centre quality analyst and extract six structured fields. LangChain's `with_structured_output(CallSummary)` enforces the schema — no manual JSON parsing.

### Structured Output

| Field | Type | Description |
|-------|------|-------------|
| `summary` | `str` | One-sentence overview |
| `customer_issue` | `str` | Primary issue raised |
| `resolution` | `str` | How it was resolved |
| `action_items` | `List[str]` | Follow-up tasks |
| `customer_sentiment` | `str` | positive / neutral / negative / frustrated |
| `key_topics` | `List[str]` | Main topics discussed |

### Error Handling

| Error | Behaviour |
|-------|-----------|
| Empty transcript | `ValueError` → FAILED |
| Missing transcript | `ValueError` → FAILED |
| Invalid API key | `AuthenticationError` → FAILED |
| API timeout | `APITimeoutError` → FAILED |
| Network error | `APIConnectionError` → FAILED |
| Structured output validation | `ValueError` → FAILED |
| Unexpected error | `RuntimeError` → FAILED |

### How to Test

```bash
# Unit tests (mocked — no API calls)
pytest tests/test_summarization_agent.py -v
```

---

## Phase 3.2 — Transcription Agent

### Purpose

The Transcription Agent is the second node in the pipeline. It reads the validated audio file from `CallState`, sends it to the OpenAI Whisper API, and writes the plain text transcript back into `CallState`.

### Input

`CallState` containing `audio_path` and `metadata` populated by the Intake Agent.

### Output

On success:

| Field | Value |
|-------|-------|
| `transcript` | Full plain-text transcript |
| `metadata.language` | Detected language (`en`) |
| `status` | `ProcessingStatus.PROCESSING` |
| `error_message` | `None` |
| `logs` | Log entry appended |

On failure:

| Field | Value |
|-------|-------|
| `status` | `ProcessingStatus.FAILED` |
| `error_message` | Human-readable description |
| `logs` | Error entry appended |

### Dependencies

| Component | Detail |
|-----------|--------|
| `WhisperService` | Wraps the OpenAI `whisper-1` API endpoint |
| `OPENAI_API_KEY` | Must be set in `.env` |

### Error Handling

| Error | Behaviour |
|-------|-----------|
| File not found | `FileNotFoundError` → FAILED |
| Empty file | `ValueError` → FAILED |
| Empty transcript | `ValueError` → FAILED |
| Invalid API key | `AuthenticationError` → FAILED |
| API timeout | `APITimeoutError` → FAILED |
| Network error | `APIConnectionError` → FAILED |
| Unexpected error | `RuntimeError` → FAILED |

### How to Test

```bash
# Unit tests (mocked — no API calls)
pytest tests/test_transcription_agent.py -v

# Full pipeline test (requires real audio file + API key)
python test_intake_manual.py
```

---

## Phase 3.1 — Call Intake Agent

### Purpose

The Call Intake Agent is the first node in the pipeline. It validates the uploaded audio file and extracts file metadata before any AI processing begins. It does not call OpenAI or Whisper.

### Input

`CallState` containing `audio_path` — the path to the uploaded audio file.

```python
state = {"audio_path": "data/audio/customer_call.mp3", "logs": []}
```

### Output

A partial `CallState` update merged by LangGraph.

On success:

| Field | Value |
|-------|-------|
| `metadata` | Populated `CallMetadata` model |
| `status` | `ProcessingStatus.PROCESSING` |
| `error_message` | `None` |
| `logs` | Log entry appended |

On failure:

| Field | Value |
|-------|-------|
| `status` | `ProcessingStatus.FAILED` |
| `error_message` | Human-readable explanation |
| `logs` | Error entry appended |

### Validation Rules

| Check | Behaviour on failure |
|-------|---------------------|
| File exists | `FileNotFoundError` → FAILED |
| Supported extension | `ValueError` → FAILED |
| File not empty | `ValueError` → FAILED |
| `audio_path` not blank | `ValueError` → FAILED |

Supported formats: `.mp3` `.wav` `.m4a` `.flac`

### Metadata Extracted

- `file_name` — original filename
- `file_type` — extension without dot (e.g. `mp3`)
- `file_size_bytes` — size in bytes
- `uploaded_at` — file creation time (UTC)
- `duration_seconds` — set to `0.0`, populated later by TranscriptionAgent
- `language` — set to `None`, detected later by TranscriptionAgent

---

## Phase 2 — Data Models & Application State

### What is Pydantic?

Pydantic is a Python library for data validation using type hints. When you create a Pydantic model, every field is validated at instantiation — wrong types, missing required fields, or out-of-range values raise a `ValidationError` immediately, before any business logic runs.

### Why Structured Models?

Without shared models, agents would pass raw dicts with no guarantees about shape or types. Pydantic models give us:

- Automatic validation — bad data is caught at the boundary, not deep inside an agent
- Clear contracts — every agent knows exactly what fields it will receive and must produce
- Serialisation — models convert cleanly to/from JSON for logging and storage
- IDE support — full autocomplete and type checking across the codebase

### What is LangGraph State?

LangGraph is a framework for building stateful multi-agent workflows as directed graphs. Each node in the graph is an agent function. Between nodes, LangGraph passes a single shared state object — `CallState` — and merges any updates the node returns back into that state.

`CallState` is defined as a `TypedDict` so that:
- LangGraph can serialise and checkpoint it natively
- Every field is typed and discoverable
- Agents can read only what they need and write only what they produce

### How Agents Communicate via CallState

```
IntakeAgent        writes → metadata, audio_path, status
       ↓ CallState
TranscriptionAgent reads  → audio_path       writes → transcript
       ↓ CallState
SummarizationAgent reads  → transcript       writes → summary
       ↓ CallState
QualityScoreAgent  reads  → transcript, summary    writes → quality_score
       ↓ CallState
RoutingAgent       reads  → summary, quality_score writes → routing_decision, status
```

Each agent receives the full `CallState`, does its work, and returns only the fields it updated. LangGraph merges those updates and passes the enriched state to the next node.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Language | Python 3.11 |
| Agent orchestration | LangGraph |
| LLM framework | LangChain + langchain-openai |
| LLM | OpenAI GPT-4o |
| Speech-to-text | OpenAI Whisper |
| Data validation | Pydantic v2 |
| UI | Streamlit |
| Config | python-dotenv |
| Token counting | tiktoken |

---

## License

MIT © 2024 CallSense-AI
