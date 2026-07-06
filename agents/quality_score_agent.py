"""
agents/quality_score_agent.py
------------------------------
Quality Score Agent — the fourth node in the CallSense-AI pipeline.

Reads the transcript and summary from CallState, calls OpenAIService
to generate a structured quality evaluation, and writes a validated
QualityScore back into CallState.
"""

from __future__ import annotations

import time
from typing import Any

from graph.state import CallState
from models.schemas import CallSummary, ProcessingStatus, QualityScore
from services.openai_service import OpenAIService
from utils.logger import get_logger

logger = get_logger(__name__)

_DIVIDER = "-" * 41


class QualityScoreAgent:
    """
    Evaluates support representative performance via GPT-4o.

    Parameters
    ----------
    openai_service : OpenAIService | None
        Optional pre-built service instance. When ``None`` a new
        ``OpenAIService`` is created using default settings.
    """

    def __init__(self, openai_service: OpenAIService | None = None) -> None:
        self._openai = openai_service or OpenAIService()
        logger.debug("QualityScoreAgent initialised.")

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def execute(self, state: CallState) -> dict[str, Any]:
        """
        Evaluate the call quality from *state*.

        Parameters
        ----------
        state : CallState
            Must contain ``transcript`` and ``summary``.

        Returns
        -------
        dict[str, Any]
            Partial state update merged by LangGraph.
            On success: ``quality_score``, ``status``, ``error_message``, ``logs``.
            On failure: ``status``, ``error_message``, ``logs``.
        """
        logs: list[str] = list(state.get("logs", []))
        logger.info("QualityScoreAgent.execute() started.")

        try:
            transcript, summary = self._validate_input(state)
            quality_score, elapsed = self._evaluate_quality(transcript, summary)
            return self._update_state(quality_score, logs, elapsed)

        except Exception as exc:
            logger.error("QualityScoreAgent failed: %s", exc, exc_info=True)
            logs.append(f"[QualityScoreAgent] ERROR: {exc}")
            return {
                "status": ProcessingStatus.FAILED,
                "error_message": str(exc),
                "logs": logs,
            }

    @staticmethod
    def format_scorecard(qs: QualityScore) -> str:
        """
        Render a ``QualityScore`` as a human-readable QA Score Card string.

        Parameters
        ----------
        qs : QualityScore
            Validated quality score model.

        Returns
        -------
        str
            Formatted scorecard suitable for display in a UI or report.
        """
        strengths = "\n".join(f"• {s}" for s in qs.strengths)
        improvements = "\n".join(f"• {a}" for a in qs.improvement_areas)

        return (
            f"📊 QUALITY SCORE CARD\n"
            f"{_DIVIDER}\n\n"
            f"Empathy\n"
            f"Score: {qs.empathy_score:.0f}/10\n"
            f"Reason: {qs.empathy_reason}\n\n"
            f"{_DIVIDER}\n\n"
            f"Professionalism\n"
            f"Score: {qs.professionalism_score:.0f}/10\n"
            f"Reason: {qs.professionalism_reason}\n\n"
            f"{_DIVIDER}\n\n"
            f"Communication Clarity\n"
            f"Score: {qs.communication_clarity_score:.0f}/10\n"
            f"Reason: {qs.communication_clarity_reason}\n\n"
            f"{_DIVIDER}\n\n"
            f"Problem Understanding\n"
            f"Score: {qs.problem_understanding_score:.0f}/10\n"
            f"Reason: {qs.problem_understanding_reason}\n\n"
            f"{_DIVIDER}\n\n"
            f"Resolution Quality\n"
            f"Score: {qs.resolution_quality_score:.0f}/10\n"
            f"Reason: {qs.resolution_quality_reason}\n\n"
            f"{_DIVIDER}\n\n"
            f"Compliance\n"
            f"Score: {qs.compliance_score:.0f}/10\n"
            f"Reason: {qs.compliance_reason}\n\n"
            f"{_DIVIDER}\n\n"
            f"Overall Score\n"
            f"{qs.overall_score:.1f} / 10\n\n"
            f"{_DIVIDER}\n\n"
            f"Strengths\n"
            f"{strengths}\n\n"
            f"{_DIVIDER}\n\n"
            f"Areas for Improvement\n"
            f"{improvements}\n\n"
            f"{_DIVIDER}\n\n"
            f"Overall Feedback\n"
            f"{qs.overall_feedback}"
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _validate_input(self, state: CallState) -> tuple[str, CallSummary]:
        """
        Validate that both transcript and summary are present in state.

        Parameters
        ----------
        state : CallState
            Current pipeline state.

        Returns
        -------
        tuple[str, CallSummary]
            Stripped transcript and validated summary.

        Raises
        ------
        ValueError
            If transcript or summary is missing or blank.
        """
        transcript = state.get("transcript", "")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError(
                "Transcript is missing or empty. "
                "Ensure the Transcription Agent ran successfully."
            )

        summary = state.get("summary")
        if not isinstance(summary, CallSummary):
            raise ValueError(
                "Summary is missing or invalid. "
                "Ensure the Summarization Agent ran successfully."
            )

        logger.debug(
            "Input validated — transcript=%d chars, summary present.", len(transcript)
        )
        return transcript.strip(), summary

    def _evaluate_quality(
        self, transcript: str, summary: CallSummary
    ) -> tuple[QualityScore, float]:
        """
        Delegate to OpenAIService and return a validated QualityScore with elapsed time.

        Parameters
        ----------
        transcript : str
            Validated transcript text.
        summary : CallSummary
            Validated summary model.

        Returns
        -------
        tuple[QualityScore, float]
            Pydantic model populated by GPT-4o and elapsed seconds.
        """
        logger.info(
            "Quality evaluation started — transcript length=%d chars, "
            "summary received (issue='%s').",
            len(transcript),
            summary.customer_issue,
        )
        start = time.perf_counter()
        quality_score = self._openai.generate_quality_score(transcript, summary)
        elapsed = time.perf_counter() - start
        logger.info(
            "Quality scores generated — overall=%.1f, elapsed=%.2fs.",
            quality_score.overall_score,
            elapsed,
        )
        return quality_score, elapsed

    def _update_state(
        self, quality_score: QualityScore, logs: list[str], elapsed: float
    ) -> dict[str, Any]:
        """
        Build the partial state update on successful evaluation.

        Parameters
        ----------
        quality_score : QualityScore
            Validated quality score model.
        logs : list[str]
            Accumulated log entries.
        elapsed : float
            Time taken to generate the evaluation in seconds.

        Returns
        -------
        dict[str, Any]
            Partial state update for LangGraph to merge.
        """
        logs.append(
            f"[QualityScoreAgent] Quality evaluation completed — "
            f"overall_score={quality_score.overall_score:.1f}, "
            f"empathy={quality_score.empathy_score:.1f}, "
            f"professionalism={quality_score.professionalism_score:.1f}, "
            f"communication_clarity={quality_score.communication_clarity_score:.1f}, "
            f"problem_understanding={quality_score.problem_understanding_score:.1f}, "
            f"resolution_quality={quality_score.resolution_quality_score:.1f}, "
            f"compliance={quality_score.compliance_score:.1f}, "
            f"strengths={len(quality_score.strengths)}, "
            f"improvement_areas={len(quality_score.improvement_areas)}, "
            f"elapsed={elapsed:.2f}s"
        )
        return {
            "quality_score": quality_score,
            "status": ProcessingStatus.PROCESSING,
            "error_message": None,
            "logs": logs,
        }
