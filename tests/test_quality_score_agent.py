"""
tests/test_quality_score_agent.py
----------------------------------
Unit tests for agents/quality_score_agent.py.

OpenAIService is mocked throughout — no real API calls are made.

Run with:
    pytest tests/test_quality_score_agent.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import openai
import pytest

from agents.quality_score_agent import QualityScoreAgent
from graph.state import CallState
from models.schemas import CallSummary, ProcessingStatus, QualityScore


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
    summary="Customer called to report a duplicate subscription charge that was resolved.",
    customer_issue="Customer was charged twice for their subscription this month.",
    resolution="Agent confirmed the duplicate charge and processed a full refund immediately.",
    action_items=["Customer should wait 3-5 business days for the refund to appear."],
    customer_sentiment="Positive",
    key_topics=["billing", "duplicate charge", "refund", "subscription"],
)

SAMPLE_QUALITY_SCORE = QualityScore(
    empathy_score=9.0,
    empathy_reason="The representative apologised and acknowledged the customer's frustration.",
    professionalism_score=10.0,
    professionalism_reason="The representative remained polite and respectful throughout.",
    communication_clarity_score=9.0,
    communication_clarity_reason="The resolution and expected timeline were explained clearly.",
    problem_understanding_score=10.0,
    problem_understanding_reason="The representative immediately identified the duplicate charge.",
    resolution_quality_score=9.0,
    resolution_quality_reason="The refund was initiated and a realistic timeline was provided.",
    compliance_score=9.0,
    compliance_reason="Standard service practices were followed throughout the interaction.",
    overall_score=9.3,
    strengths=["Strong empathy", "Clear communication", "Effective resolution"],
    improvement_areas=["Could proactively confirm refund timeline updates."],
    overall_feedback=(
        "The representative handled the interaction professionally and empathetically. "
        "The customer's issue was correctly identified and resolved with a clear explanation "
        "and expected timeline. Overall, this was a high-quality customer service interaction."
    ),
)


@pytest.fixture
def mock_openai() -> MagicMock:
    svc = MagicMock()
    svc.generate_quality_score.return_value = SAMPLE_QUALITY_SCORE
    return svc


@pytest.fixture
def agent(mock_openai: MagicMock) -> QualityScoreAgent:
    return QualityScoreAgent(openai_service=mock_openai)


@pytest.fixture
def valid_state() -> CallState:
    return {
        "transcript": SAMPLE_TRANSCRIPT,
        "summary": SAMPLE_SUMMARY,
        "status": ProcessingStatus.PROCESSING,
        "logs": ["[IntakeAgent] OK", "[TranscriptionAgent] OK", "[SummarizationAgent] OK"],
    }


# ── Successful evaluation ──────────────────────────────────────────────────────

class TestSuccessfulEvaluation:
    def test_status_is_processing(self, agent: QualityScoreAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert result["status"] == ProcessingStatus.PROCESSING

    def test_quality_score_is_correct_type(self, agent: QualityScoreAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert isinstance(result["quality_score"], QualityScore)

    def test_all_score_fields_present(self, agent: QualityScoreAgent, valid_state: CallState) -> None:
        qs = agent.execute(valid_state)["quality_score"]
        assert qs.empathy_score == 9.0
        assert qs.professionalism_score == 10.0
        assert qs.communication_clarity_score == 9.0
        assert qs.problem_understanding_score == 10.0
        assert qs.resolution_quality_score == 9.0
        assert qs.compliance_score == 9.0
        assert qs.overall_score == 9.3

    def test_all_reason_fields_non_empty(self, agent: QualityScoreAgent, valid_state: CallState) -> None:
        qs = agent.execute(valid_state)["quality_score"]
        assert len(qs.empathy_reason) > 0
        assert len(qs.professionalism_reason) > 0
        assert len(qs.communication_clarity_reason) > 0
        assert len(qs.problem_understanding_reason) > 0
        assert len(qs.resolution_quality_reason) > 0
        assert len(qs.compliance_reason) > 0

    def test_scores_within_range(self, agent: QualityScoreAgent, valid_state: CallState) -> None:
        qs = agent.execute(valid_state)["quality_score"]
        for score in [
            qs.empathy_score, qs.professionalism_score, qs.communication_clarity_score,
            qs.problem_understanding_score, qs.resolution_quality_score,
            qs.compliance_score, qs.overall_score,
        ]:
            assert 0.0 <= score <= 10.0

    def test_strengths_non_empty_list(self, agent: QualityScoreAgent, valid_state: CallState) -> None:
        qs = agent.execute(valid_state)["quality_score"]
        assert isinstance(qs.strengths, list) and len(qs.strengths) >= 1

    def test_improvement_areas_non_empty_list(self, agent: QualityScoreAgent, valid_state: CallState) -> None:
        qs = agent.execute(valid_state)["quality_score"]
        assert isinstance(qs.improvement_areas, list) and len(qs.improvement_areas) >= 1

    def test_overall_feedback_non_empty(self, agent: QualityScoreAgent, valid_state: CallState) -> None:
        qs = agent.execute(valid_state)["quality_score"]
        assert isinstance(qs.overall_feedback, str) and len(qs.overall_feedback) > 0

    def test_error_message_is_none(self, agent: QualityScoreAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert result["error_message"] is None

    def test_log_contains_completion_message(self, agent: QualityScoreAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert any("Quality evaluation completed" in log for log in result["logs"])

    def test_log_contains_overall_score(self, agent: QualityScoreAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert any("overall_score" in log for log in result["logs"])

    def test_log_contains_elapsed(self, agent: QualityScoreAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert any("elapsed" in log for log in result["logs"])

    def test_previous_logs_preserved(self, agent: QualityScoreAgent, valid_state: CallState) -> None:
        result = agent.execute(valid_state)
        assert result["logs"][0] == "[IntakeAgent] OK"
        assert result["logs"][1] == "[TranscriptionAgent] OK"
        assert result["logs"][2] == "[SummarizationAgent] OK"

    def test_openai_called_with_transcript_and_summary(
        self, agent: QualityScoreAgent, mock_openai: MagicMock, valid_state: CallState
    ) -> None:
        agent.execute(valid_state)
        mock_openai.generate_quality_score.assert_called_once_with(
            SAMPLE_TRANSCRIPT.strip(), SAMPLE_SUMMARY
        )


# ── Scorecard formatting ───────────────────────────────────────────────────────

class TestFormatScorecard:
    def test_returns_string(self) -> None:
        card = QualityScoreAgent.format_scorecard(SAMPLE_QUALITY_SCORE)
        assert isinstance(card, str)

    def test_contains_header(self) -> None:
        card = QualityScoreAgent.format_scorecard(SAMPLE_QUALITY_SCORE)
        assert "QUALITY SCORE CARD" in card

    def test_contains_all_dimension_labels(self) -> None:
        card = QualityScoreAgent.format_scorecard(SAMPLE_QUALITY_SCORE)
        for label in [
            "Empathy", "Professionalism", "Communication Clarity",
            "Problem Understanding", "Resolution Quality", "Compliance",
        ]:
            assert label in card

    def test_contains_overall_score(self) -> None:
        card = QualityScoreAgent.format_scorecard(SAMPLE_QUALITY_SCORE)
        assert "Overall Score" in card
        assert "9.3" in card

    def test_contains_strengths_section(self) -> None:
        card = QualityScoreAgent.format_scorecard(SAMPLE_QUALITY_SCORE)
        assert "Strengths" in card
        assert "• Strong empathy" in card

    def test_contains_improvement_areas_section(self) -> None:
        card = QualityScoreAgent.format_scorecard(SAMPLE_QUALITY_SCORE)
        assert "Areas for Improvement" in card
        assert "•" in card

    def test_contains_overall_feedback_section(self) -> None:
        card = QualityScoreAgent.format_scorecard(SAMPLE_QUALITY_SCORE)
        assert "Overall Feedback" in card
        assert SAMPLE_QUALITY_SCORE.overall_feedback in card

    def test_contains_reason_text(self) -> None:
        card = QualityScoreAgent.format_scorecard(SAMPLE_QUALITY_SCORE)
        assert SAMPLE_QUALITY_SCORE.empathy_reason in card
        assert SAMPLE_QUALITY_SCORE.compliance_reason in card


# ── Missing transcript ─────────────────────────────────────────────────────────

class TestMissingTranscript:
    def test_empty_transcript_fails(self, agent: QualityScoreAgent, valid_state: CallState) -> None:
        valid_state["transcript"] = ""
        result = agent.execute(valid_state)
        assert result["status"] == ProcessingStatus.FAILED

    def test_whitespace_transcript_fails(self, agent: QualityScoreAgent, valid_state: CallState) -> None:
        valid_state["transcript"] = "   "
        result = agent.execute(valid_state)
        assert result["status"] == ProcessingStatus.FAILED

    def test_missing_transcript_key_fails(self, agent: QualityScoreAgent) -> None:
        state: CallState = {"summary": SAMPLE_SUMMARY, "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED

    def test_error_message_mentions_transcript(self, agent: QualityScoreAgent) -> None:
        state: CallState = {"transcript": "", "summary": SAMPLE_SUMMARY, "logs": []}
        result = agent.execute(state)
        assert "transcript" in result["error_message"].lower()


# ── Missing summary ────────────────────────────────────────────────────────────

class TestMissingSummary:
    def test_none_summary_fails(self, agent: QualityScoreAgent) -> None:
        state: CallState = {"transcript": SAMPLE_TRANSCRIPT, "summary": None, "logs": []}  # type: ignore
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED

    def test_missing_summary_key_fails(self, agent: QualityScoreAgent) -> None:
        state: CallState = {"transcript": SAMPLE_TRANSCRIPT, "logs": []}
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED

    def test_error_message_mentions_summary(self, agent: QualityScoreAgent) -> None:
        state: CallState = {"transcript": SAMPLE_TRANSCRIPT, "summary": None, "logs": []}  # type: ignore
        result = agent.execute(state)
        assert "summary" in result["error_message"].lower()


# ── OpenAI API exceptions ──────────────────────────────────────────────────────

class TestOpenAIExceptions:
    def _make_agent(self, side_effect: Exception) -> QualityScoreAgent:
        svc = MagicMock()
        svc.generate_quality_score.side_effect = side_effect
        return QualityScoreAgent(openai_service=svc)

    def _state(self) -> CallState:
        return {"transcript": SAMPLE_TRANSCRIPT, "summary": SAMPLE_SUMMARY, "logs": []}

    def test_authentication_error(self) -> None:
        agent = self._make_agent(
            openai.AuthenticationError("Invalid key", response=MagicMock(), body={})
        )
        assert agent.execute(self._state())["status"] == ProcessingStatus.FAILED

    def test_timeout_error(self) -> None:
        agent = self._make_agent(openai.APITimeoutError(request=MagicMock()))
        assert agent.execute(self._state())["status"] == ProcessingStatus.FAILED

    def test_connection_error(self) -> None:
        agent = self._make_agent(openai.APIConnectionError(request=MagicMock()))
        assert agent.execute(self._state())["status"] == ProcessingStatus.FAILED

    def test_runtime_error(self) -> None:
        agent = self._make_agent(RuntimeError("Unexpected GPT failure"))
        result = agent.execute(self._state())
        assert result["status"] == ProcessingStatus.FAILED
        assert result["error_message"] is not None


# ── Validation failure ─────────────────────────────────────────────────────────

class TestValidationFailure:
    def test_structured_output_validation_error_fails(self) -> None:
        svc = MagicMock()
        svc.generate_quality_score.side_effect = ValueError(
            "Structured output validation failed."
        )
        agent = QualityScoreAgent(openai_service=svc)
        state: CallState = {
            "transcript": SAMPLE_TRANSCRIPT,
            "summary": SAMPLE_SUMMARY,
            "logs": [],
        }
        result = agent.execute(state)
        assert result["status"] == ProcessingStatus.FAILED
        assert "validation" in result["error_message"].lower()
