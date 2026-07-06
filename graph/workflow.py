"""
graph/workflow.py
-----------------
LangGraph StateGraph definition for CallSense-AI.

This module wires together all five agents into a directed graph:

    intake → transcription → summary → quality_score → routing → END

Each node is a thin wrapper that delegates to the corresponding agent
class. Conditional edges allow the graph to short-circuit to an error
terminal node if any agent sets ``state["error"]``.

Usage
-----
    from graph.workflow import build_graph

    app = build_graph()
    result = app.invoke(initial_state)
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from agents.intake_agent import CallIntakeAgent
from agents.quality_score_agent import QualityScoreAgent
from agents.routing_agent import RoutingAgent
from agents.summarization_agent import SummarizationAgent
from agents.transcription_agent import TranscriptionAgent
from graph.state import CallState
from utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
# Agent singletons — instantiated once when the module is imported
# ------------------------------------------------------------------ #
_intake_agent = CallIntakeAgent()
_transcription_agent = TranscriptionAgent()
_summarization_agent = SummarizationAgent()
_quality_score_agent = QualityScoreAgent()
_routing_agent = RoutingAgent()


# ------------------------------------------------------------------ #
# Node functions — each delegates to the matching agent
# ------------------------------------------------------------------ #

def intake_node(state: CallState) -> dict[str, Any]:
    """
    LangGraph node: Call Intake.

    Validates the incoming audio file and populates initial metadata.
    Sets ``state["error"]`` on failure.
    """
    logger.info("[Node] intake_node — call_id=%s", state.get("call_id"))
    # TODO: delegate to _intake_agent.execute(state) once implemented
    return {}


def transcription_node(state: CallState) -> dict[str, Any]:
    """
    LangGraph node: Transcription.

    Sends the audio file to WhisperService and stores the raw transcript.
    Sets ``state["error"]`` on failure.
    """
    logger.info("[Node] transcription_node — call_id=%s", state.get("call_id"))
    # TODO: delegate to _transcription_agent.execute(state) once implemented
    return {}


def summary_node(state: CallState) -> dict[str, Any]:
    """
    LangGraph node: Summarisation.

    Sends the transcript to GPT-4o and stores the structured summary.
    Sets ``state["error"]`` on failure.
    """
    logger.info("[Node] summary_node — call_id=%s", state.get("call_id"))
    # TODO: delegate to _summarization_agent.execute(state) once implemented
    return {}


def quality_score_node(state: CallState) -> dict[str, Any]:
    """
    LangGraph node: Quality Scoring.

    Evaluates the transcript and summary against QA criteria.
    Sets ``state["error"]`` on failure.
    """
    logger.info("[Node] quality_score_node — call_id=%s", state.get("call_id"))
    # TODO: delegate to _quality_score_agent.execute(state) once implemented
    return {}


def routing_node(state: CallState) -> dict[str, Any]:
    """
    LangGraph node: Routing.

    Decides the next action (resolve, escalate, callback, etc.)
    based on the summary and QA score.
    Sets ``state["error"]`` on failure.
    """
    logger.info("[Node] routing_node — call_id=%s", state.get("call_id"))
    # TODO: delegate to _routing_agent.execute(state) once implemented
    return {}


def error_node(state: CallState) -> dict[str, Any]:
    """
    LangGraph terminal node: Error handler.

    Logs the error and marks the call as failed. This node is reached
    via conditional edges whenever ``state["error"]`` is non-None.
    """
    logger.error(
        "[Node] error_node — call_id=%s | error=%s",
        state.get("call_id"),
        state.get("error"),
    )
    # TODO: persist error record, trigger alerts, etc.
    return {"status": "failed"}


# ------------------------------------------------------------------ #
# Conditional edge helpers
# ------------------------------------------------------------------ #

def _has_error(state: CallState) -> str:
    """
    Return ``"error"`` if the state contains an error, otherwise ``"ok"``.

    Used as the condition function for all conditional edges so that
    any failing node immediately routes to ``error_node``.
    """
    return "error" if state.get("error") else "ok"


# ------------------------------------------------------------------ #
# Graph builder
# ------------------------------------------------------------------ #

def build_graph() -> Any:
    """
    Construct and compile the CallSense-AI LangGraph workflow.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph application ready to be invoked with
        an initial ``CallState`` dict.

    Graph topology
    --------------
    START
      └─► intake ──(error?)──► error_node ──► END
                └─(ok)──► transcription ──(error?)──► error_node
                              └─(ok)──► summary ──(error?)──► error_node
                                          └─(ok)──► quality_score ──(error?)──► error_node
                                                        └─(ok)──► routing ──► END
    """
    graph = StateGraph(CallState)

    # ---------------------------------------------------------------- #
    # Register nodes
    # ---------------------------------------------------------------- #
    graph.add_node("intake", intake_node)
    graph.add_node("transcription", transcription_node)
    graph.add_node("summary", summary_node)
    graph.add_node("quality_score", quality_score_node)
    graph.add_node("routing", routing_node)
    graph.add_node("error_node", error_node)

    # ---------------------------------------------------------------- #
    # Entry point
    # ---------------------------------------------------------------- #
    graph.set_entry_point("intake")

    # ---------------------------------------------------------------- #
    # Conditional edges — route to error_node on any failure
    # ---------------------------------------------------------------- #
    graph.add_conditional_edges(
        "intake",
        _has_error,
        {"ok": "transcription", "error": "error_node"},
    )
    graph.add_conditional_edges(
        "transcription",
        _has_error,
        {"ok": "summary", "error": "error_node"},
    )
    graph.add_conditional_edges(
        "summary",
        _has_error,
        {"ok": "quality_score", "error": "error_node"},
    )
    graph.add_conditional_edges(
        "quality_score",
        _has_error,
        {"ok": "routing", "error": "error_node"},
    )

    # ---------------------------------------------------------------- #
    # Terminal edges
    # ---------------------------------------------------------------- #
    graph.add_edge("routing", END)
    graph.add_edge("error_node", END)

    logger.info("CallSense-AI workflow graph compiled successfully.")
    return graph.compile()
