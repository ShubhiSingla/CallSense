"""
tests/test_transcription_agent.py
----------------------------------
Unit tests for agents/transcription_agent.py.

WhisperService is mocked throughout — no real API calls are made.

Run with:
    pytest tests/test_transcription_agent.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import openai
import pytest

from agents.transcription_agent import TranscriptionAgent
from graph.state import CallState
from models.schemas import CallMetadata, ProcessingStatus


# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_TRANSCRIPT = "Thank you for calling. How can I help you today?"


@pytest.fixture
def mock_whisper() -> MagicMock:
    """WhisperService mock that returns a valid transcript."""
    svc = MagicMock()
    svc.transcribe.return_value = SAMPLE_TRANSCRIPT
    return svc


@pytest.fixture
def agent(mock_whisper: MagicMock) -> TranscriptionAgent:
    return TranscriptionAgent(whisper_service=mock_whisper)


@pytest.fixture
def valid_audio(tmp_path: Path) -> Path:
    f = tmp_path / "call.wav"
    f.write_bytes(b"\x00" * 1024)
    return f


@pytest.fixture
def valid_state(valid_audio: Path) -> CallState:
    metadata = CallMetadata(
        file_name="call.wav",
        file_type="wav",
        file_size_bytes=1024,
        duration_seconds=0.0,
    )
    return {
        "audio_path": str(valid_audio),
        "metadata": metadata,
        "status": ProcessingStatus.PROCESSING,
        "logs": ["[IntakeAgent] OK — file=call.wav, size=1024 bytes"],
    }


# ── Successful transcription ───────────────────────────────────────────────────

class TestSuccessfulTranscription:
    def test_status_is_processing(self, agent: TranscriptionAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert result["status"] == ProcessingStatus.PROCESSING

    def test_transcript_is_populated(self, agent: TranscriptionAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert result["transcript"] == SAMPLE_TRANSCRIPT

    def test_error_message_is_none(self, agent: TranscriptionAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert result["error_message"] is None

    def test_log_entry_appended(self, agent: TranscriptionAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert any("TranscriptionAgent" in log for log in result["logs"])

    def test_previous_logs_preserved(self, agent: TranscriptionAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert result["logs"][0] == "[IntakeAgent] OK — file=call.wav, size=1024 bytes"

    def test_metadata_language_updated(self, agent: TranscriptionAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert result["metadata"].language == "en"

    def test_whisper_called_with_correct_path(
        self, agent: TranscriptionAgent, mock_whisper: MagicMock, valid_state: CallState
    ) -> None:
        agent.execute(valid_state)
        mock_whisper.transcribe.assert_called_once()


# ── Missing file ───────────────────────────────────────────────────────────────

class TestMissingFile:
    def test_status_is_failed(self, agent: TranscriptionAgent) -> None:
        state: CallState = {"audio_path": "/nonexistent/call.wav", "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED

    def test_error_message_set(self, agent: TranscriptionAgent) -> None:
        state: CallState = {"audio_path": "/nonexistent/call.wav", "logs": []}
        result = agent.execute(state)
        assert "not found" in result["error_message"].lower()

    def test_empty_path_fails(self, agent: TranscriptionAgent) -> None:
        state: CallState = {"audio_path": "", "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED


# ── Empty file ─────────────────────────────────────────────────────────────────

class TestEmptyFile:
    def test_empty_file_fails(self, agent: TranscriptionAgent, tmp_path: Path) -> None:
        f = tmp_path / "empty.wav"
        f.write_bytes(b"")
        state: CallState = {"audio_path": str(f), "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED

    def test_empty_file_error_message(self, agent: TranscriptionAgent, tmp_path: Path) -> None:
        f = tmp_path / "empty.mp3"
        f.write_bytes(b"")
        state: CallState = {"audio_path": str(f), "logs": []}
        result = agent.execute(state)
        assert "empty" in result["error_message"].lower()


# ── Empty transcript ───────────────────────────────────────────────────────────

class TestEmptyTranscript:
    def test_empty_transcript_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "call.wav"
        f.write_bytes(b"\x00" * 512)

        whisper = MagicMock()
        whisper.transcribe.return_value = "   "  # blank after strip
        agent = TranscriptionAgent(whisper_service=whisper)

        # WhisperService raises ValueError for empty transcript —
        # simulate that behaviour in the mock
        whisper.transcribe.side_effect = ValueError("Whisper returned an empty transcript.")

        state: CallState = {"audio_path": str(f), "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED
        assert "empty" in result["error_message"].lower()


# ── Whisper API exceptions ─────────────────────────────────────────────────────

class TestWhisperAPIExceptions:
    def _make_agent(self, side_effect: Exception) -> TranscriptionAgent:
        whisper = MagicMock()
        whisper.transcribe.side_effect = side_effect
        return TranscriptionAgent(whisper_service=whisper)

    def _state(self, tmp_path: Path) -> CallState:
        f = tmp_path / "call.wav"
        f.write_bytes(b"\x00" * 512)
        return {"audio_path": str(f), "logs": []}

    def test_authentication_error(self, tmp_path: Path) -> None:
        agent = self._make_agent(
            openai.AuthenticationError("Invalid key", response=MagicMock(), body={})
        )
        result = agent.execute(self._state(tmp_path))
        assert result["status"] == ProcessingStatus.FAILED

    def test_timeout_error(self, tmp_path: Path) -> None:
        agent = self._make_agent(openai.APITimeoutError(request=MagicMock()))
        result = agent.execute(self._state(tmp_path))
        assert result["status"] == ProcessingStatus.FAILED

    def test_connection_error(self, tmp_path: Path) -> None:
        agent = self._make_agent(openai.APIConnectionError(request=MagicMock()))
        result = agent.execute(self._state(tmp_path))
        assert result["status"] == ProcessingStatus.FAILED

    def test_generic_runtime_error(self, tmp_path: Path) -> None:
        agent = self._make_agent(RuntimeError("Unexpected Whisper failure"))
        result = agent.execute(self._state(tmp_path))
        assert result["status"] == ProcessingStatus.FAILED
        assert result["error_message"] is not None
