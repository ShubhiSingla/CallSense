"""
tests/test_intake.py
--------------------
Unit tests for CallIntakeAgent.

These tests verify the intake agent's validation and metadata
extraction logic in isolation, without touching any external
services or the LangGraph workflow.

Run with:
    pytest tests/test_intake.py -v
"""

from __future__ import annotations

import pytest

# TODO: Import CallIntakeAgent once implemented
# from agents.intake_agent import CallIntakeAgent
# from utils.validator import ValidationError


class TestCallIntakeAgentInit:
    """Tests for CallIntakeAgent.__init__."""

    def test_agent_initialises_without_error(self) -> None:
        """CallIntakeAgent should instantiate without raising."""
        # TODO: agent = CallIntakeAgent()
        # TODO: assert agent is not None
        pytest.skip("CallIntakeAgent not yet implemented.")


class TestCallIntakeAgentExecute:
    """Tests for CallIntakeAgent.execute()."""

    def test_execute_returns_call_id_on_valid_audio(self, tmp_path) -> None:
        """
        Given a valid audio file, execute() should return a state dict
        containing a non-empty call_id string.
        """
        # TODO: Create a small dummy .wav file in tmp_path.
        # TODO: agent = CallIntakeAgent()
        # TODO: result = agent.execute({"audio_path": str(dummy_wav)})
        # TODO: assert "call_id" in result
        # TODO: assert result["call_id"] != ""
        pytest.skip("CallIntakeAgent not yet implemented.")

    def test_execute_sets_error_on_missing_file(self) -> None:
        """
        Given a path that does not exist, execute() should return a
        state dict with a non-None error field.
        """
        # TODO: agent = CallIntakeAgent()
        # TODO: result = agent.execute({"audio_path": "/nonexistent/file.wav"})
        # TODO: assert result["error"] is not None
        # TODO: assert result["status"] == "failed"
        pytest.skip("CallIntakeAgent not yet implemented.")

    def test_execute_sets_error_on_unsupported_format(self, tmp_path) -> None:
        """
        Given a file with an unsupported extension, execute() should
        return a state dict with a non-None error field.
        """
        # TODO: Create a dummy .txt file in tmp_path.
        # TODO: agent = CallIntakeAgent()
        # TODO: result = agent.execute({"audio_path": str(dummy_txt)})
        # TODO: assert result["error"] is not None
        pytest.skip("CallIntakeAgent not yet implemented.")

    def test_execute_populates_metadata(self, tmp_path) -> None:
        """
        Given a valid audio file, execute() should return a metadata
        dict containing at least filename, file_size_bytes, and extension.
        """
        # TODO: Create a small dummy .wav file in tmp_path.
        # TODO: agent = CallIntakeAgent()
        # TODO: result = agent.execute({"audio_path": str(dummy_wav)})
        # TODO: assert "filename" in result["metadata"]
        # TODO: assert "file_size_bytes" in result["metadata"]
        pytest.skip("CallIntakeAgent not yet implemented.")


class TestCallIntakeAgentHelpers:
    """Tests for CallIntakeAgent private helper methods."""

    def test_generate_call_id_returns_unique_values(self) -> None:
        """
        _generate_call_id() should return a different UUID on each call.
        """
        # TODO: agent = CallIntakeAgent()
        # TODO: id1 = agent._generate_call_id()
        # TODO: id2 = agent._generate_call_id()
        # TODO: assert id1 != id2
        pytest.skip("CallIntakeAgent not yet implemented.")
