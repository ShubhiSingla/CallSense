"""
examples/sample_data.py
-----------------------
Realistic sample instances of all CallSense-AI Pydantic models.

Use these for manual testing, UI prototyping, and as fixtures
in unit tests.
"""

from __future__ import annotations

from datetime import datetime

from models.schemas import CallMetadata, CallSummary, QualityScore, RoutingDecision

sample_metadata = CallMetadata(
    file_name="support_call_20240715.mp3",
    file_type="audio/mpeg",
    file_size_bytes=4_823_040,
    duration_seconds=187.5,
    language="en-US",
    uploaded_at=datetime(2024, 7, 15, 9, 30, 0),
)

sample_summary = CallSummary(
    summary="Customer called to report an incorrect charge on their monthly invoice.",
    customer_issue="Customer was billed twice for the Premium plan in June 2024.",
    resolution="Agent confirmed the duplicate charge and raised a refund request (REF-00482).",
    action_items=[
        "Process refund of $49.99 within 3–5 business days",
        "Send confirmation email to customer",
        "Flag billing system for duplicate-charge audit",
    ],
    customer_sentiment="frustrated",
    key_topics=["billing", "duplicate charge", "refund", "Premium plan"],
)

sample_quality_score = QualityScore(
    empathy_score=8.5,
    professionalism_score=9.0,
    resolution_score=8.0,
    compliance_score=9.5,
    overall_score=8.75,
    feedback=(
        "Agent demonstrated strong empathy and resolved the issue efficiently. "
        "Compliance phrases were used correctly. Consider offering a courtesy "
        "discount on the next invoice to improve customer satisfaction."
    ),
)

sample_routing_decision = RoutingDecision(
    status="escalated",
    next_agent="Billing Specialist Team",
    reason=(
        "Duplicate charge requires manual verification by the billing team "
        "before the refund can be processed."
    ),
    retry_count=0,
)
