"""
agents/quality_score_agent.py
------------------------------
Quality Score Agent — evaluates call quality using GPT-4o.

Responsibilities
----------------
- Receive the transcript and summary from the state.
- Invoke ``OpenAIService.score_call()`` to get a structured QA score.
- Validate the score values are within acceptable ranges.
- Store the QA score dict in the state.
- Handle LLM errors gracefully and set ``state["error"]``.

This agent is the only component that calls ``OpenAIService.score_call()``.
"""

from __future__ import annotations

from typing import Any

from config.settings import settings
from graph.state import CallState
from services.openai_service import OpenAIService
from utils.logger import get_logger
from utils.validator import validate_qa_score

logger = get_logger(__name__)

# Required keys that must be present in the LLM's JSON response
_REQUIRED_QA_KEYS: frozenset[str] = frozenset(
    {
        "overall_score",
        "empathy_score",
        "resolution_score",
        "communication_score",
        "compliance_passed",
        "feedback",
    }
)


class QualityScoreAgent:
    """
    Evaluates the quality of a customer support call using GPT-4o.

    Scores are produced on a 0.0 – 10.0 scale across three dimensions
    (empathy, resolution, communication) plus a boolean compliance flag.

    Parameters
    ----------
    openai_service : OpenAIService | None
        Optional pre-built service instance. When ``None`` a new
        ``OpenAIService`` is created using default settings.
    pass_threshold : float
        Minimum ``overall_score`` to consider a call as passing QA.
        Defaults to ``settings.QA_SCORE_PASS_THRESHOLD``.
    """

    def __init__(
        self,
        openai_service: OpenAIService | None = None,
        pass_threshold: float = settings.QA_SCORE_PASS_THRESHOLD,
    ) -> None:
        self._openai = openai_service or OpenAIService()
        self._pass_threshold = pass_threshold
        logger.debug(
            "QualityScoreAgent initialised (pass_threshold=%.1f).",
            self._pass_threshold,
        )

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def execute(self, state: CallState) -> dict[str, Any]:
        """
        Score the call using the transcript and summary in *state*.

        Parameters
        ----------
        state : CallState
            Current graph state. Must contain ``"transcript"`` and
            ``"summary"``.

        Returns
        -------
        dict[str, Any]
            Partial state update dict.
            On success: ``qa_score`` (dict).
            On failure: ``error``, ``status``.
        """
        logger.info(
            "QualityScoreAgent.execute() — call_id=%s", state.get("call_id")
        )

        try:
            transcript = state.get("transcript", "")
            summary = state.get("summary", {})

            self._validate_inputs(transcript, summary)
            qa_score = self._score_call(transcript, summary)
            self._validate_score_schema(qa_score)
            passed = self._check_pass_threshold(qa_score)

            logger.info(
                "QA scoring complete — overall=%.1f, passed=%s",
                qa_score.get("overall_score", 0.0),
                passed,
            )
            return {"qa_score": qa_score, "error": None}

        except Exception as exc:
            logger.error("QualityScoreAgent failed: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _validate_inputs(
        self, transcript: str, summary: dict[str, Any]
    ) -> None:
        """
        Ensure both transcript and summary are present before scoring.

        Parameters
        ----------
        transcript : str
            Raw transcript text.
        summary : dict[str, Any]
            Structured summary dict.

        Raises
        ------
        ValueError
            If either input is empty or missing.
        """
        # TODO: Raise ValueError if transcript is empty.
        # TODO: Raise ValueError if summary is empty or missing required keys.
        raise NotImplementedError("_validate_inputs() is not yet implemented.")

    def _score_call(
        self, transcript: str, summary: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Invoke ``OpenAIService.score_call()`` and return the QA score dict.

        Parameters
        ----------
        transcript : str
            Validated transcript text.
        summary : dict[str, Any]
            Structured summary dict.

        Returns
        -------
        dict[str, Any]
            QA score dict matching ``models.schemas.QualityScore`` fields.
        """
        logger.debug("Calling OpenAIService.score_call().")
        # TODO: return self._openai.score_call(transcript, summary)
        raise NotImplementedError("_score_call() is not yet implemented.")

    def _validate_score_schema(self, qa_score: dict[str, Any]) -> None:
        """
        Verify that the LLM response contains all required QA keys and
        that numeric scores are within [0.0, 10.0].

        Parameters
        ----------
        qa_score : dict[str, Any]
            Parsed QA score dict from the LLM.

        Raises
        ------
        ValueError
            If any required key is missing or a score is out of range.
        """
        missing = _REQUIRED_QA_KEYS - qa_score.keys()
        if missing:
            raise ValueError(f"QA score response missing keys: {missing}")

        # TODO: Call validate_qa_score() for each numeric score field.
        raise NotImplementedError("_validate_score_schema() is not yet implemented.")

    def _check_pass_threshold(self, qa_score: dict[str, Any]) -> bool:
        """
        Return ``True`` if the overall score meets the pass threshold.

        Parameters
        ----------
        qa_score : dict[str, Any]
            Validated QA score dict.

        Returns
        -------
        bool
        """
        # TODO: return qa_score.get("overall_score", 0.0) >= self._pass_threshold
        raise NotImplementedError("_check_pass_threshold() is not yet implemented.")
