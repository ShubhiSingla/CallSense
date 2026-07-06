"""
tests/test_summarization_agent.py
----------------------------------
Unit tests for agents/summarization_agent.py.

OpenAIService is mocked throughout — no real API calls are made.

Run with:
    pytest tests/test_summarization_agent.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import openai
import pytest

from agents.summarization_agent import SummarizationAgent
from graph.state import CallState
from models.schemas import CallSummary, ProcessingStatus


# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_TRANSCRIPT = (
    "Agent: Thank you for calling support. How can I help you today?\n"
    "Customer: I was charged twice for my subscription this month.\n"
    "Agent: I'm sorry to hear that. Let me look into this for you.\n"
    "Agent: I can confirm the duplicate charge and will process a refund immediately.\n"
    "Customer: Thank you, I appreciate that.\n"
    "Agent: The refund will appear within 3-5 business days. Is there anything else?\n"
    "Customer: No, that's all. Thank you.\n"
)

SAMPLE_SUMMARY = CallSummary(
    summary=(
        "The customer contacted support to report a duplicate charge on their subscription. "
        "The agent reviewed the account, confirmed the error, and immediately initiated a refund. "
        "The customer was informed the refund would appear within 3-5 business days. "
        "The customer expressed satisfaction with the resolution."
    ),
    customer_issue="Customer was charged twice for their subscription this month.",
    resolution="Agent confirmed the duplicate charge and processed a full refund immediately.",
    action_items=[
        "Customer should wait 3-5 business days for the refund to appear.",
        "Send a confirmation email to the customer once the refund is processed.",
    ],
    customer_sentiment="Positive",
    key_topics=["billing", "duplicate charge", "refund", "subscription", "customer support"],
)


@pytest.fixture
def mock_openai() -> MagicMock:
    svc = MagicMock()
    svc.generate_summary.return_value = SAMPLE_SUMMARY
    return svc


@pytest.fixture
def agent(mock_openai: MagicMock) -> SummarizationAgent:
    return SummarizationAgent(openai_service=mock_openai)


@pytest.fixture
def valid_state() -> CallState:
    return {
        "transcript": SAMPLE_TRANSCRIPT,
        "status": ProcessingStatus.PROCESSING,
        "logs": ["[IntakeAgent] OK", "[TranscriptionAgent] OK"],
    }


# ── Successful summarisation ───────────────────────────────────────────────────

class TestSuccessfulSummary:
    def test_status_is_processing(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert result["status"] == ProcessingStatus.PROCESSING

    def test_summary_is_callsummary_instance(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert isinstance(result["summary"], CallSummary)

    def test_customer_issue_extracted(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert result["summary"].customer_issue == SAMPLE_SUMMARY.customer_issue

    def test_resolution_extracted(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert result["summary"].resolution == SAMPLE_SUMMARY.resolution

    def test_action_items_non_empty(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert len(result["summary"].action_items) >= 1

    def test_action_items_are_strings(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert all(isinstance(item, str) for item in result["summary"].action_items)

    def test_sentiment_is_valid(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert result["summary"].customer_sentiment in {"Positive", "Neutral", "Negative"}

    def test_sentiment_value(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert result["summary"].customer_sentiment == "Positive"

    def test_key_topics_count(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert 3 <= len(result["summary"].key_topics) <= 5

    def test_key_topics_are_strings(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert all(isinstance(t, str) for t in result["summary"].key_topics)

    def test_summary_is_detailed(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        """Executive summary should be multi-sentence (at least 50 chars)."""
        result = agent.execute(valid_state)
        assert len(result["summary"].summary) >= 50

    def test_error_message_is_none(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert result["error_message"] is None

    def test_log_contains_summary_generated(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert any("Summary generated successfully" in log for log in result["logs"])

    def test_log_contains_customer_issue(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert any("customer_issue" in log for log in result["logs"])

    def test_log_contains_sentiment(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert any("sentiment" in log for log in result["logs"])

    def test_log_contains_elapsed(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert any("elapsed" in log for log in result["logs"])

    def test_previous_logs_preserved(self, agent: SummarizationAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert result["logs"][0] == "[IntakeAgent] OK"
        assert result["logs"][1] == "[TranscriptionAgent] OK"

    def test_openai_called_with_transcript(
        self, agent: SummarizationAgent, mock_openai: MagicMock, valid_state: CallState
    ) -> None:
        agent.execute(valid_state)
        mock_openai.generate_summary.assert_called_once_with(SAMPLE_TRANSCRIPT.strip())


# ── Empty transcript ───────────────────────────────────────────────────────────

class TestEmptyTranscript:
    def test_empty_string_fails(self, agent: SummarizationAgent) -> None:
        state: CallState = {"transcript": "", "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED

    def test_whitespace_only_fails(self, agent: SummarizationAgent) -> None:
        state: CallState = {"transcript": "   ", "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED

    def test_error_message_mentions_transcript(self, agent: SummarizationAgent) -> None:
        state: CallState = {"transcript": "", "logs": []}
        result = agent.execute(state)
        assert "transcript" in result["error_message"].lower()


# ── Missing transcript ─────────────────────────────────────────────────────────

class TestMissingTranscript:
    def test_missing_key_fails(self, agent: SummarizationAgent) -> None:
        state: CallState = {"logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED

    def test_none_transcript_fails(self, agent: SummarizationAgent) -> None:
        state: CallState = {"transcript": None, "logs": []}  # type: ignore
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED


# ── OpenAI API exceptions ──────────────────────────────────────────────────────

class TestOpenAIExceptions:
    def _make_agent(self, side_effect: Exception) -> SummarizationAgent:
        svc = MagicMock()
        svc.generate_summary.side_effect = side_effect
        return SummarizationAgent(openai_service=svc)

    def _state(self) -> CallState:
        return {"transcript": SAMPLE_TRANSCRIPT, "logs": []}

    def test_authentication_error(self) -> None:
        agent = self._make_agent(
            openai.AuthenticationError("Invalid key", response=MagicMock(), body={})
        )
        result = agent.execute(self._state())
        assert result["status"] == ProcessingStatus.FAILED

    def test_timeout_error(self) -> None:
        agent = self._make_agent(openai.APITimeoutError(request=MagicMock()))
        result = agent.execute(self._state())
        assert result["status"] == ProcessingStatus.FAILED

    def test_connection_error(self) -> None:
        agent = self._make_agent(openai.APIConnectionError(request=MagicMock()))
        result = agent.execute(self._state())
        assert result["status"] == ProcessingStatus.FAILED

    def test_runtime_error(self) -> None:
        agent = self._make_agent(RuntimeError("Unexpected GPT failure"))
        result = agent.execute(self._state())
        assert result["status"] == ProcessingStatus.FAILED
        assert result["error_message"] is not None


# ── Invalid structured output ──────────────────────────────────────────────────

class TestInvalidStructuredOutput:
    def test_validation_error_fails(self) -> None:
        svc = MagicMock()
        # Return a plain string instead of a CallSummary — simulates bad output
        svc.generate_summary.side_effect = ValueError("Structured output validation failed.")
        agent = SummarizationAgent(openai_service=svc)
        state: CallState = {"transcript": SAMPLE_TRANSCRIPT, "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED
        assert "validation" in result["error_message"].lower()
