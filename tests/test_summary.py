"""
tests/test_summary.py
---------------------
Unit tests for SummarizationAgent.

These tests verify schema validation and error-handling logic
in isolation using a mocked OpenAIService so no real API calls
are made during the test suite.

Run with:
    pytest tests/test_summary.py -v
"""

from __future__ import annotations

import pytest

# TODO: Import SummarizationAgent and OpenAIService once implemented
# from agents.summarization_agent import SummarizationAgent
# from services.openai_service import OpenAIService
# from unittest.mock import MagicMock


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

SAMPLE_TRANSCRIPT = (
    "Agent: Thank you for calling support, how can I help you today? "
    "Customer: Hi, my internet has been down for two days. "
    "Agent: I'm sorry to hear that. Let me look into your account. "
    "Customer: Please hurry, I work from home. "
    "Agent: I can see there's an outage in your area. It should be resolved "
    "within 4 hours. I'll also apply a credit to your account. "
    "Customer: Thank you, I appreciate that. "
    "Agent: Is there anything else I can help you with? "
    "Customer: No, that's all. "
    "Agent: Have a great day!"
)

VALID_SUMMARY = {
    "brief": "Customer reported a two-day internet outage; agent confirmed area outage and applied credit.",
    "key_issues": ["Internet outage lasting two days"],
    "resolution": "Agent confirmed area outage resolving in 4 hours and applied account credit.",
    "action_items": ["Monitor outage resolution", "Confirm credit applied"],
    "sentiment": "neutral",
}


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestSummarizationAgentInit:
    """Tests for SummarizationAgent.__init__."""

    def test_agent_initialises_without_error(self) -> None:
        """SummarizationAgent should instantiate without raising."""
        # TODO: agent = SummarizationAgent()
        # TODO: assert agent is not None
        pytest.skip("SummarizationAgent not yet implemented.")


class TestSummarizationAgentExecute:
    """Tests for SummarizationAgent.execute()."""

    def test_execute_returns_summary_on_valid_transcript(self) -> None:
        """
        Given a valid transcript and a mocked OpenAIService that returns
        VALID_SUMMARY, execute() should return a state dict with
        ``summary`` populated and ``error`` as None.
        """
        # TODO: mock_openai = MagicMock()
        # TODO: mock_openai.summarize.return_value = VALID_SUMMARY
        # TODO: agent = SummarizationAgent(openai_service=mock_openai)
        # TODO: result = agent.execute({"transcript": SAMPLE_TRANSCRIPT})
        # TODO: assert result["summary"] == VALID_SUMMARY
        # TODO: assert result["error"] is None
        pytest.skip("SummarizationAgent not yet implemented.")

    def test_execute_sets_error_on_empty_transcript(self) -> None:
        """
        Given an empty transcript, execute() should return a state dict
        with a non-None error field without calling the LLM.
        """
        # TODO: mock_openai = MagicMock()
        # TODO: agent = SummarizationAgent(openai_service=mock_openai)
        # TODO: result = agent.execute({"transcript": ""})
        # TODO: assert result["error"] is not None
        # TODO: mock_openai.summarize.assert_not_called()
        pytest.skip("SummarizationAgent not yet implemented.")

    def test_execute_sets_error_on_missing_summary_keys(self) -> None:
        """
        If the LLM returns a dict missing required keys, execute() should
        return a state dict with a non-None error field.
        """
        # TODO: mock_openai = MagicMock()
        # TODO: mock_openai.summarize.return_value = {"brief": "only brief"}
        # TODO: agent = SummarizationAgent(openai_service=mock_openai)
        # TODO: result = agent.execute({"transcript": SAMPLE_TRANSCRIPT})
        # TODO: assert result["error"] is not None
        pytest.skip("SummarizationAgent not yet implemented.")

    def test_execute_sets_error_on_llm_exception(self) -> None:
        """
        If OpenAIService.summarize() raises an exception, execute() should
        catch it and return a state dict with a non-None error field.
        """
        # TODO: mock_openai = MagicMock()
        # TODO: mock_openai.summarize.side_effect = RuntimeError("API error")
        # TODO: agent = SummarizationAgent(openai_service=mock_openai)
        # TODO: result = agent.execute({"transcript": SAMPLE_TRANSCRIPT})
        # TODO: assert result["error"] is not None
        # TODO: assert result["status"] == "failed"
        pytest.skip("SummarizationAgent not yet implemented.")


class TestSummarizationAgentValidation:
    """Tests for SummarizationAgent._validate_summary_schema()."""

    def test_valid_summary_passes_schema_check(self) -> None:
        """A complete summary dict should not raise."""
        # TODO: agent = SummarizationAgent.__new__(SummarizationAgent)
        # TODO: agent._validate_summary_schema(VALID_SUMMARY)  # should not raise
        pytest.skip("SummarizationAgent not yet implemented.")

    def test_incomplete_summary_raises_value_error(self) -> None:
        """A summary dict missing required keys should raise ValueError."""
        # TODO: agent = SummarizationAgent.__new__(SummarizationAgent)
        # TODO: with pytest.raises(ValueError):
        # TODO:     agent._validate_summary_schema({"brief": "only brief"})
        pytest.skip("SummarizationAgent not yet implemented.")
