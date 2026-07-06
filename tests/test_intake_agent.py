"""
tests/test_intake_agent.py
--------------------------
Unit tests for agents/intake_agent.py.

Run with:
    pytest tests/test_intake_agent.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.intake_agent import CallIntakeAgent, SUPPORTED_EXTENSIONS
from graph.state import CallState
from models.schemas import ProcessingStatus


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def agent() -> CallIntakeAgent:
    return CallIntakeAgent()


@pytest.fixture
def valid_wav(tmp_path: Path) -> Path:
    """A minimal non-empty .wav file."""
    f = tmp_path / "call.wav"
    f.write_bytes(b"\x00" * 1024)
    return f


@pytest.fixture
def valid_mp3(tmp_path: Path) -> Path:
    f = tmp_path / "call.mp3"
    f.write_bytes(b"\x00" * 1024)
    return f


# ── Initialisation ─────────────────────────────────────────────────────────────

class TestInit:
    def test_instantiates_without_error(self, agent: CallIntakeAgent) -> None:
        assert agent is not None


# ── Valid audio ────────────────────────────────────────────────────────────────

class TestValidAudio:
    def test_status_is_processing(self, agent: CallIntakeAgent, valid_wav: Path) -> None:
        state: CallState = {"audio_path": str(valid_wav), "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.PROCESSING

    def test_metadata_is_populated(self, agent: CallIntakeAgent, valid_wav: Path) -> None:
        state: CallState = {"audio_path": str(valid_wav), "logs": []}
        result = agent.execute(state)
        metadata = result["metadata"]
        assert metadata.file_name == "call.wav"
        assert metadata.file_size_bytes == 1024
        assert metadata.file_type == "wav"

    def test_error_message_is_none(self, agent: CallIntakeAgent, valid_wav: Path) -> None:
        state: CallState = {"audio_path": str(valid_wav), "logs": []}
        result = agent.execute(state)
        assert result["error_message"] is None

    def test_log_entry_added(self, agent: CallIntakeAgent, valid_wav: Path) -> None:
        state: CallState = {"audio_path": str(valid_wav), "logs": []}
        result = agent.execute(state)
        assert len(result["logs"]) == 1
        assert "OK" in result["logs"][0]

    def test_mp3_accepted(self, agent: CallIntakeAgent, valid_mp3: Path) -> None:
        state: CallState = {"audio_path": str(valid_mp3), "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.PROCESSING

    def test_all_supported_extensions_accepted(
        self, agent: CallIntakeAgent, tmp_path: Path
    ) -> None:
        for ext in SUPPORTED_EXTENSIONS:
            f = tmp_path / f"call{ext}"
            f.write_bytes(b"\x00" * 512)
            state: CallState = {"audio_path": str(f), "logs": []}
            result = agent.execute(state)
            assert result["status"] == ProcessingStatus.PROCESSING, f"Failed for {ext}"


# ── Missing file ───────────────────────────────────────────────────────────────

class TestMissingFile:
    def test_status_is_failed(self, agent: CallIntakeAgent) -> None:
        state: CallState = {"audio_path": "/nonexistent/call.wav", "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED

    def test_error_message_is_set(self, agent: CallIntakeAgent) -> None:
        state: CallState = {"audio_path": "/nonexistent/call.wav", "logs": []}
        result = agent.execute(state)
        assert result["error_message"] is not None
        assert "not found" in result["error_message"].lower()

    def test_empty_path_fails(self, agent: CallIntakeAgent) -> None:
        state: CallState = {"audio_path": "", "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED
        assert result["error_message"] is not None


# ── Unsupported extension ──────────────────────────────────────────────────────

class TestUnsupportedExtension:
    def test_txt_file_fails(self, agent: CallIntakeAgent, tmp_path: Path) -> None:
        f = tmp_path / "call.txt"
        f.write_bytes(b"hello")
        state: CallState = {"audio_path": str(f), "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED

    def test_error_mentions_unsupported(self, agent: CallIntakeAgent, tmp_path: Path) -> None:
        f = tmp_path / "call.pdf"
        f.write_bytes(b"hello")
        state: CallState = {"audio_path": str(f), "logs": []}
        result = agent.execute(state)
        assert "unsupported" in result["error_message"].lower()

    def test_mp4_accepted(self, agent: CallIntakeAgent, tmp_path: Path) -> None:
        """mp4 is supported (added at user request)."""
        f = tmp_path / "call.mp4"
        f.write_bytes(b"\x00" * 512)
        state: CallState = {"audio_path": str(f), "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.PROCESSING


# ── Empty file ─────────────────────────────────────────────────────────────────

class TestEmptyFile:
    def test_empty_file_fails(self, agent: CallIntakeAgent, tmp_path: Path) -> None:
        f = tmp_path / "empty.wav"
        f.write_bytes(b"")
        state: CallState = {"audio_path": str(f), "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED

    def test_empty_file_error_message(self, agent: CallIntakeAgent, tmp_path: Path) -> None:
        f = tmp_path / "empty.mp3"
        f.write_bytes(b"")
        state: CallState = {"audio_path": str(f), "logs": []}
        result = agent.execute(state)
        assert "empty" in result["error_message"].lower()


# ── Logs preserved ─────────────────────────────────────────────────────────────

class TestLogsPreserved:
    def test_existing_logs_are_kept(self, agent: CallIntakeAgent, valid_wav: Path) -> None:
        state: CallState = {"audio_path": str(valid_wav), "logs": ["[PreviousAgent] done"]}
        result = agent.execute(state)
        assert result["logs"][0] == "[PreviousAgent] done"
        assert len(result["logs"]) == 2
