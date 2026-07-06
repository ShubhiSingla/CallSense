"""
graph/state.py
--------------
LangGraph shared state for CallSense-AI.

``CallState`` is the single object that flows through every node in
the LangGraph workflow. Each agent reads from and writes to this state;
LangGraph merges updates automatically between nodes.
"""

from __future__ import annotations

from typing import List, Optional
from typing_extensions import TypedDict

from models.schemas import (
    CallMetadata,
    CallSummary,
    ProcessingStatus,
    QualityScore,
    RoutingDecision,
)


class CallState(TypedDict, total=False):
    """
    Shared state passed between all LangGraph nodes.

    Fields
    ------
    metadata : CallMetadata | None
        Audio file information populated by the Intake Agent.
    audio_path : str
        Absolute path to the uploaded audio file on disk.
    transcript : str
        Raw transcript text produced by the Transcription Agent.
    summary : CallSummary | None
        Structured summary produced by the Summarization Agent.
    quality_score : QualityScore | None
        QA evaluation produced by the Quality Score Agent.
    routing_decision : RoutingDecision | None
        Routing outcome produced by the Routing Agent.
    status : ProcessingStatus
        Current pipeline lifecycle status.
    error_message : str | None
        Error detail set by any node that raises an exception.
    logs : List[str]
        Ordered list of log messages emitted by each agent.
    """

    metadata: Optional[CallMetadata]
    audio_path: str
    transcript: str
    summary: Optional[CallSummary]
    quality_score: Optional[QualityScore]
    routing_decision: Optional[RoutingDecision]
    status: ProcessingStatus
    error_message: Optional[str]
    logs: List[str]
