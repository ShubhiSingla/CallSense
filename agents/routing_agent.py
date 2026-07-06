"""
agents/routing_agent.py
-----------------------
Routing Agent — the final decision-maker in the CallSense-AI pipeline.

Responsibilities
----------------
- Receive the summary and QA score from the state.
- Apply rule-based pre-checks (e.g. very low QA score → escalate).
- Invoke ``OpenAIService.decide_routing()`` for nuanced decisions.
- Store the routing decision dict in the state.
- Mark the call status as ``"completed"`` or ``"escalated"``.
- Handle errors gracefully and set ``state["error"]``.

This agent is the terminal processing node before the Streamlit UI
receives the final result.
"""

from __future__ import annotations

from typing import Any

from config.settings import settings
from graph.state import CallState
from services.openai_service import OpenAIService
from utils.logger import get_logger

logger = get_logger(__name__)

# QA score below this value triggers an immediate escalation
# without consulting the LLM, saving API cost.
_HARD_ESCALATION_THRESHOLD: float = 3.0


class RoutingAgent:
    """
    Determines the next action for a completed call.

    Routing decisions include: resolved, escalate to human agent,
    escalate to supervisor, schedule callback, or send follow-up email.

    Parameters
    ----------
    openai_service : OpenAIService | None
        Optional pre-built service instance. When ``None`` a new
        ``OpenAIService`` is created using default settings.
    """

    def __init__(self, openai_service: OpenAIService | None = None) -> None:
        self._openai = openai_service or OpenAIService()
        logger.debug("RoutingAgent initialised.")

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def execute(self, state: CallState) -> dict[str, Any]:
        """
        Determine the routing action for the call in *state*.

        Parameters
        ----------
        state : CallState
            Current graph state. Must contain ``"summary"`` and
            ``"qa_score"``.

        Returns
        -------
        dict[str, Any]
            Partial state update dict.
            On success: ``routing_decision`` (dict), ``status``.
            On failure: ``error``, ``status``.
        """
        logger.info(
            "RoutingAgent.execute() — call_id=%s", state.get("call_id")
        )

        try:
            summary = state.get("summary", {})
            qa_score = state.get("qa_score", {})

            self._validate_inputs(summary, qa_score)

            # Fast-path: hard escalation for very low QA scores
            if self._requires_hard_escalation(qa_score):
                decision = self._build_hard_escalation_decision(qa_score)
            else:
                decision = self._llm_routing_decision(summary, qa_score)

            final_status = self._determine_final_status(decision)

            logger.info(
                "Routing complete — decision=%s, status=%s",
                decision.get("decision"),
                final_status,
            )
            return {
                "routing_decision": decision,
                "status": final_status,
                "error": None,
            }

        except Exception as exc:
            logger.error("RoutingAgent failed: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _validate_inputs(
        self, summary: dict[str, Any], qa_score: dict[str, Any]
    ) -> None:
        """
        Ensure summary and qa_score are present before routing.

        Raises
        ------
        ValueError
            If either input is empty.
        """
        # TODO: Raise ValueError if summary is empty.
        # TODO: Raise ValueError if qa_score is empty.
        raise NotImplementedError("_validate_inputs() is not yet implemented.")

    def _requires_hard_escalation(self, qa_score: dict[str, Any]) -> bool:
        """
        Return ``True`` if the QA score is so low that immediate escalation
        is warranted without consulting the LLM.

        Parameters
        ----------
        qa_score : dict[str, Any]
            Validated QA score dict.

        Returns
        -------
        bool
        """
        # TODO: return qa_score.get("overall_score", 10.0) < _HARD_ESCALATION_THRESHOLD
        raise NotImplementedError(
            "_requires_hard_escalation() is not yet implemented."
        )

    def _build_hard_escalation_decision(
        self, qa_score: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Build a pre-canned escalation decision dict without calling the LLM.

        Parameters
        ----------
        qa_score : dict[str, Any]
            QA score dict that triggered the hard escalation.

        Returns
        -------
        dict[str, Any]
            Routing decision dict matching ``models.schemas.RoutingDecision``.
        """
        # TODO: Return a dict with decision="escalate_supervisor", priority=5,
        #       and a reason explaining the low QA score.
        raise NotImplementedError(
            "_build_hard_escalation_decision() is not yet implemented."
        )

    def _llm_routing_decision(
        self,
        summary: dict[str, Any],
        qa_score: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Ask GPT-4o to determine the routing action.

        Parameters
        ----------
        summary : dict[str, Any]
            Structured call summary.
        qa_score : dict[str, Any]
            QA score dict.

        Returns
        -------
        dict[str, Any]
            Routing decision dict from the LLM.
        """
        logger.debug("Calling OpenAIService.decide_routing().")
        # TODO: return self._openai.decide_routing(summary, qa_score)
        raise NotImplementedError("_llm_routing_decision() is not yet implemented.")

    def _determine_final_status(self, decision: dict[str, Any]) -> str:
        """
        Map a routing decision to a final call status string.

        Parameters
        ----------
        decision : dict[str, Any]
            Routing decision dict.

        Returns
        -------
        str
            One of ``"completed"`` or ``"escalated"``.
        """
        # TODO: Return "escalated" if decision["decision"] starts with "escalate",
        #       otherwise return "completed".
        raise NotImplementedError("_determine_final_status() is not yet implemented.")
