"""
tests/test_models.py
--------------------
Unit tests for models/schemas.py and graph/state.py.

Run with:
    pytest tests/test_models.py -v
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.schemas import (
    CallMetadata,
    CallSummary,
    ProcessingStatus,
    QualityScore,
    RoutingDecision,
)
from graph.state import CallState


# ── ProcessingStatus ───────────────────────────────────────────────────────────

class TestProcessingStatus:
    def test_all_values_exist(self):
        assert ProcessingStatus.PENDING == "PENDING"
        assert ProcessingStatus.PROCESSING == "PROCESSING"
        assert ProcessingStatus.COMPLETED == "COMPLETED"
        assert ProcessingStatus.FAILED == "FAILED"

    def test_is_string_enum(self):
        assert isinstance(ProcessingStatus.PENDING, str)


# ── CallMetadata ───────────────────────────────────────────────────────────────

class TestCallMetadata:
    def test_valid_creation(self):
        m = CallMetadata(
            file_name="call.mp3",
            file_type="audio/mpeg",
            file_size_bytes=1024,
            duration_seconds=60.0,
        )
        assert m.file_name == "call.mp3"
        assert m.language is None

    def test_with_language(self):
        m = CallMetadata(
            file_name="call.wav",
            file_type="audio/wav",
            file_size_bytes=2048,
            duration_seconds=120.0,
            language="en-US",
        )
        assert m.language == "en-US"

    def test_negative_file_size_fails(self):
        with pytest.raises(ValidationError):
            CallMetadata(
                file_name="call.mp3",
                file_type="audio/mpeg",
                file_size_bytes=-1,
                duration_seconds=60.0,
            )

    def test_negative_duration_fails(self):
        with pytest.raises(ValidationError):
            CallMetadata(
                file_name="call.mp3",
                file_type="audio/mpeg",
                file_size_bytes=1024,
                duration_seconds=-5.0,
            )

    def test_missing_required_field_fails(self):
        with pytest.raises(ValidationError):
            CallMetadata(file_type="audio/mpeg", file_size_bytes=1024, duration_seconds=60.0)


# ── CallSummary ────────────────────────────────────────────────────────────────

class TestCallSummary:
    def test_valid_creation(self):
        s = CallSummary(
            summary="Customer reported a billing issue.",
            customer_issue="Duplicate charge on invoice.",
            resolution="Refund raised.",
            customer_sentiment="frustrated",
        )
        assert s.action_items == []
        assert s.key_topics == []

    def test_with_lists(self):
        s = CallSummary(
            summary="Overview.",
            customer_issue="Issue.",
            resolution="Resolved.",
            customer_sentiment="positive",
            action_items=["Send email"],
            key_topics=["billing"],
        )
        assert len(s.action_items) == 1
        assert len(s.key_topics) == 1

    def test_missing_required_field_fails(self):
        with pytest.raises(ValidationError):
            CallSummary(customer_issue="Issue.", resolution="Done.", customer_sentiment="neutral")


# ── QualityScore ───────────────────────────────────────────────────────────────

class TestQualityScore:
    def test_valid_creation(self):
        q = QualityScore(
            empathy_score=8.0,
            empathy_reason="Agent apologised for the inconvenience.",
            professionalism_score=9.0,
            professionalism_reason="Tone was polite throughout.",
            communication_clarity_score=8.5,
            communication_clarity_reason="Explanations were clear and concise.",
            problem_understanding_score=9.0,
            problem_understanding_reason="Issue was identified immediately.",
            resolution_quality_score=7.5,
            resolution_quality_reason="Issue resolved with a clear next step.",
            compliance_score=9.5,
            compliance_reason="All standard practices were followed.",
            overall_score=8.5,
            strengths=["Clear communication", "Empathetic tone"],
            improvement_areas=["Could confirm resolution more explicitly."],
            overall_feedback="Good call.",
        )
        assert q.overall_score == 8.5

    def test_score_above_max_fails(self):
        with pytest.raises(ValidationError):
            QualityScore(
                empathy_score=11.0,
                empathy_reason="Too high.",
                professionalism_score=9.0,
                professionalism_reason="Fine.",
                communication_clarity_score=8.5,
                communication_clarity_reason="Fine.",
                problem_understanding_score=9.0,
                problem_understanding_reason="Fine.",
                resolution_quality_score=7.5,
                resolution_quality_reason="Fine.",
                compliance_score=9.5,
                compliance_reason="Fine.",
                overall_score=8.5,
                strengths=["Good"],
                improvement_areas=["Improve X."],
                overall_feedback="Good call.",
            )

    def test_score_below_min_fails(self):
        with pytest.raises(ValidationError):
            QualityScore(
                empathy_score=-1.0,
                empathy_reason="Too low.",
                professionalism_score=9.0,
                professionalism_reason="Fine.",
                communication_clarity_score=8.5,
                communication_clarity_reason="Fine.",
                problem_understanding_score=9.0,
                problem_understanding_reason="Fine.",
                resolution_quality_score=7.5,
                resolution_quality_reason="Fine.",
                compliance_score=9.5,
                compliance_reason="Fine.",
                overall_score=8.5,
                strengths=["Good"],
                improvement_areas=["Improve X."],
                overall_feedback="Good call.",
            )


# ── RoutingDecision ────────────────────────────────────────────────────────────

class TestRoutingDecision:
    def test_valid_creation(self):
        r = RoutingDecision(
            status="escalated",
            next_agent="Billing Team",
            reason="Requires manual review.",
        )
        assert r.retry_count == 0

    def test_custom_retry_count(self):
        r = RoutingDecision(
            status="resolved",
            next_agent="None",
            reason="Issue resolved on first contact.",
            retry_count=2,
        )
        assert r.retry_count == 2

    def test_negative_retry_count_fails(self):
        with pytest.raises(ValidationError):
            RoutingDecision(
                status="resolved",
                next_agent="None",
                reason="Done.",
                retry_count=-1,
            )

    def test_missing_required_field_fails(self):
        with pytest.raises(ValidationError):
            RoutingDecision(status="resolved", reason="Done.")


# ── CallState ──────────────────────────────────────────────────────────────────

class TestCallState:
    def test_minimal_state_initialisation(self):
        state: CallState = {
            "audio_path": "data/audio/call.mp3",
            "transcript": "",
            "status": ProcessingStatus.PENDING,
            "error_message": None,
            "logs": [],
        }
        assert state["status"] == ProcessingStatus.PENDING
        assert state["logs"] == []

    def test_state_with_all_fields(self):
        metadata = CallMetadata(
            file_name="call.mp3",
            file_type="audio/mpeg",
            file_size_bytes=1024,
            duration_seconds=60.0,
        )
        state: CallState = {
            "metadata": metadata,
            "audio_path": "data/audio/call.mp3",
            "transcript": "Hello, how can I help?",
            "summary": None,
            "quality_score": None,
            "routing_decision": None,
            "status": ProcessingStatus.PROCESSING,
            "error_message": None,
            "logs": ["Intake complete."],
        }
        assert state["metadata"].file_name == "call.mp3"
        assert state["status"] == ProcessingStatus.PROCESSING
        assert len(state["logs"]) == 1
