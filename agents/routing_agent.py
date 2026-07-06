"""
agents/routing_agent.py
------------------------
Routing Agent — the fifth and final node in the CallSense-AI pipeline.

Reads the summary and quality_score from CallState and applies
deterministic business rules to produce a RoutingDecision.

No LLM is used. All decisions are rule-based.
"""

from __future__ import annotations

import time
from typing import Any

from graph.state import CallState
from models.schemas import (
    CallSummary,
    ProcessingStatus,
    QualityScore,
    RoutingDecision,
    RoutingOutcome,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Routing rule thresholds ────────────────────────────────────────────────────

_COMPLETED_MIN_SCORE: float = 9.0
_MANUAL_QA_MIN_SCORE: float = 7.0
_COMPLIANCE_MAX_SCORE: float = 5.0
_ESCALATE_MAX_RESOLUTION: float = 6.0

# Human-readable team names per outcome
_NEXT_AGENT: dict[RoutingOutcome, str] = {
    RoutingOutcome.COMPLETED: "None — call closed",
    RoutingOutcome.MANUAL_QA_REVIEW: "QA Team",
    RoutingOutcome.SUPERVISOR_REVIEW: "Supervisor",
    RoutingOutcome.ESCALATE: "Escalation Team",
    RoutingOutcome.COMPLIANCE_REVIEW: "Compliance Team",
    RoutingOutcome.CUSTOMER_FOLLOW_UP: "Customer Success Team",
}


class RoutingAgent:
    """
    Applies deterministic business rules to route a completed call.

    No external services or LLMs are used — all decisions are derived
    from the ``QualityScore`` and ``CallSummary`` already in ``CallState``.
    """

    def __init__(self) -> None:
        logger.debug("RoutingAgent initialised.")

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def execute(self, state: CallState) -> dict[str, Any]:
        """
        Determine the routing decision for the call in *state*.

        Parameters
        ----------
        state : CallState
            Must contain ``summary`` and ``quality_score``.

        Returns
        -------
        dict[str, Any]
            Partial state update merged by LangGraph.
            On success: ``routing_decision``, ``status``, ``error_message``, ``logs``.
            On failure: ``status``, ``error_message``, ``logs``.
        """
        logs: list[str] = list(state.get("logs", []))
        logger.info("RoutingAgent.execute() started.")

        try:
            summary, quality_score = self._validate_input(state)
            routing_decision, elapsed = self._determine_route(summary, quality_score)
            return self._update_state(routing_decision, logs, elapsed)

        except Exception as exc:
            logger.error("RoutingAgent failed: %s", exc, exc_info=True)
            logs.append(f"[RoutingAgent] ERROR: {exc}")
            return {
                "status": ProcessingStatus.FAILED,
                "error_message": str(exc),
                "logs": logs,
            }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _validate_input(
        self, state: CallState
    ) -> tuple[CallSummary, QualityScore]:
        """
        Validate that summary and quality_score are present in state.

        Parameters
        ----------
        state : CallState
            Current pipeline state.

        Returns
        -------
        tuple[CallSummary, QualityScore]
            Validated summary and quality score.

        Raises
        ------
        ValueError
            If either field is missing or of the wrong type.
        """
        summary = state.get("summary")
        if not isinstance(summary, CallSummary):
            raise ValueError(
                "Summary is missing or invalid. "
                "Ensure the Summarization Agent ran successfully."
            )

        quality_score = state.get("quality_score")
        if not isinstance(quality_score, QualityScore):
            raise ValueError(
                "Quality score is missing or invalid. "
                "Ensure the Quality Score Agent ran successfully."
            )

        logger.debug(
            "Input validated — overall_score=%.1f, sentiment=%s.",
            quality_score.overall_score,
            summary.customer_sentiment,
        )
        return summary, quality_score

    def _determine_route(
        self, summary: CallSummary, qs: QualityScore
    ) -> tuple[RoutingDecision, float]:
        """
        Apply business rules and return a RoutingDecision with elapsed time.

        Rules are evaluated in priority order — the first matching rule wins.

        Parameters
        ----------
        summary : CallSummary
            Validated summary from the Summarization Agent.
        qs : QualityScore
            Validated quality score from the Quality Score Agent.

        Returns
        -------
        tuple[RoutingDecision, float]
            The routing decision and elapsed seconds.
        """
        start = time.perf_counter()

        overall = qs.overall_score
        sentiment = summary.customer_sentiment.lower()
        resolution_quality = qs.resolution_quality_score
        compliance = qs.compliance_score
        resolution_text = summary.resolution.lower()

        logger.info(
            "Routing evaluation — overall_score=%.1f, sentiment=%s, "
            "resolution_quality=%.1f, compliance=%.1f.",
            overall, sentiment, resolution_quality, compliance,
        )

        outcome, reason = self._apply_rules(
            overall, sentiment, resolution_quality, compliance, resolution_text
        )

        elapsed = time.perf_counter() - start
        logger.info(
            "Routing decision — outcome=%s, elapsed=%.4fs.", outcome.value, elapsed
        )

        decision = RoutingDecision(
            outcome=outcome,
            next_agent=_NEXT_AGENT[outcome],
            reason=reason,
        )
        return decision, elapsed

    @staticmethod
    def _apply_rules(
        overall: float,
        sentiment: str,
        resolution_quality: float,
        compliance: float,
        resolution_text: str,
    ) -> tuple[RoutingOutcome, str]:
        """
        Evaluate all routing rules in priority order and return the first match.

        Priority order:
          1. Compliance Review  (compliance failure is highest risk)
          2. Escalation         (negative sentiment + poor resolution)
          3. Customer Follow-up (pending resolution)
          4. Supervisor Review  (low overall score)
          5. Manual QA Review   (mid-range overall score)
          6. Completed          (high score + positive/neutral sentiment)

        Parameters
        ----------
        overall : float
            Overall quality score.
        sentiment : str
            Lowercased customer sentiment string.
        resolution_quality : float
            Resolution quality score.
        compliance : float
            Compliance score.
        resolution_text : str
            Lowercased resolution field from the summary.

        Returns
        -------
        tuple[RoutingOutcome, str]
            Matched outcome and human-readable reason.
        """
        # Rule 5 — Compliance Review (highest priority)
        if compliance <= _COMPLIANCE_MAX_SCORE:
            return (
                RoutingOutcome.COMPLIANCE_REVIEW,
                f"Compliance score is {compliance:.1f}/10, which is at or below the "
                f"acceptable threshold of {_COMPLIANCE_MAX_SCORE:.0f}/10. "
                "This call requires immediate compliance review.",
            )

        # Rule 4 — Escalation
        if sentiment in ("negative", "frustrated") and resolution_quality <= _ESCALATE_MAX_RESOLUTION:
            return (
                RoutingOutcome.ESCALATE,
                f"Customer sentiment is {sentiment} and the resolution quality score "
                f"is {resolution_quality:.1f}/10, indicating the issue was not "
                "adequately resolved. Escalation is required.",
            )

        # Rule 6 — Customer Follow-up
        if "pending" in resolution_text:
            return (
                RoutingOutcome.CUSTOMER_FOLLOW_UP,
                "The call resolution is marked as pending. "
                "A follow-up with the customer is required to close the issue.",
            )

        # Rule 3 — Supervisor Review
        if overall < _MANUAL_QA_MIN_SCORE:
            return (
                RoutingOutcome.SUPERVISOR_REVIEW,
                f"Overall quality score is {overall:.1f}/10, which is below the "
                f"acceptable threshold of {_MANUAL_QA_MIN_SCORE:.0f}/10. "
                "This call requires supervisor review.",
            )

        # Rule 2 — Manual QA Review
        if overall < _COMPLETED_MIN_SCORE:
            return (
                RoutingOutcome.MANUAL_QA_REVIEW,
                f"Overall quality score is {overall:.1f}/10. "
                "The call meets minimum standards but requires manual QA review "
                "before it can be marked as completed.",
            )

        # Rule 1 — Completed
        return (
            RoutingOutcome.COMPLETED,
            f"Overall quality score is {overall:.1f}/10 and customer sentiment is "
            f"{sentiment}. The call was handled to a high standard and is marked "
            "as completed.",
        )

    def _update_state(
        self,
        routing_decision: RoutingDecision,
        logs: list[str],
        elapsed: float,
    ) -> dict[str, Any]:
        """
        Build the partial state update on successful routing.

        Parameters
        ----------
        routing_decision : RoutingDecision
            The determined routing decision.
        logs : list[str]
            Accumulated log entries.
        elapsed : float
            Time taken to determine the route in seconds.

        Returns
        -------
        dict[str, Any]
            Partial state update for LangGraph to merge.
        """
        logs.append(
            f"[RoutingAgent] Routing completed — "
            f"outcome={routing_decision.outcome.value}, "
            f"next_agent='{routing_decision.next_agent}', "
            f"elapsed={elapsed:.4f}s"
        )
        return {
            "routing_decision": routing_decision,
            "status": ProcessingStatus.COMPLETED,
            "error_message": None,
            "logs": logs,
        }
