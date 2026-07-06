"""
test_intake_manual.py
---------------------
Manual end-to-end test for Phase 3.1 + 3.2.

Chains CallIntakeAgent → TranscriptionAgent against a real audio file.

Run with:
    python test_intake_manual.py
"""

from agents.intake_agent import CallIntakeAgent
from agents.transcription_agent import TranscriptionAgent
from models.schemas import ProcessingStatus

AUDIO_PATH = "data/audio/sample.wav"  # ← change to your audio file


def divider(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


# ── Step 1: Intake Agent ───────────────────────────────────────────────────────

divider("STEP 1 — Call Intake Agent")
state = {"audio_path": AUDIO_PATH, "logs": []}

intake_result = CallIntakeAgent().execute(state)
state.update(intake_result)

print(f"STATUS  : {state.get('status')}")
print(f"ERROR   : {state.get('error_message')}")

if state.get("metadata"):
    m = state["metadata"]
    print(f"FILE    : {m.file_name}")
    print(f"TYPE    : {m.file_type}")
    print(f"SIZE    : {m.file_size_bytes} bytes")
    print(f"UPLOAD  : {m.uploaded_at}")

# Stop here if intake failed
if state.get("status") == ProcessingStatus.FAILED:
    print("\n❌ Intake failed — skipping transcription.")
    exit(1)

# ── Step 2: Transcription Agent ────────────────────────────────────────────────

divider("STEP 2 — Transcription Agent  (calling Whisper API...)")

transcription_result = TranscriptionAgent().execute(state)
state.update(transcription_result)

print(f"STATUS  : {state.get('status')}")
print(f"ERROR   : {state.get('error_message')}")

if state.get("transcript"):
    transcript = state["transcript"]
    print(f"WORDS   : {len(transcript.split())}")
    print(f"LANG    : {state['metadata'].language}")
    print(f"\nTRANSCRIPT PREVIEW:\n{transcript[:300]}{'...' if len(transcript) > 300 else ''}")

# ── Logs ───────────────────────────────────────────────────────────────────────

divider("AGENT LOGS")
for entry in state.get("logs", []):
    print(entry)
