"""
agents/summarization_agent.py
------------------------------
Summarization Agent — the third node in the CallSense-AI pipeline.

Reads the transcript from CallState, calls OpenAIService to generate
a structured summary, and writes a validated CallSummary back into
CallState.
"""

from __future__ import annotations

import time
from typing import Any

from graph.state import CallState
from models.schemas import CallSummary, ProcessingStatus
from services.openai_service import OpenAIService
from utils.logger import get_logger

logger = get_logger(__name__)


class SummarizationAgent:
    """
    Generates a structured CallSummary from a call transcript via GPT-4o.

    Parameters
    ----------
    openai_service : OpenAIService | None
        Optional pre-built service instance. When ``None`` a new
        ``OpenAIService`` is created using default settings.
    """

    def __init__(self, openai_service: OpenAIService | None = None) -> None:
        self._openai = openai_service or OpenAIService()
        logger.debug("SummarizationAgent initialised.")

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def execute(self, state: CallState) -> dict[str, Any]:
        """
        Summarise the transcript stored in *state*.

        Parameters
        ----------
        state : CallState
            Must contain ``transcript``. Should also contain ``logs``.

        Returns
        -------
        dict[str, Any]
            Partial state update merged by LangGraph.
            On success: ``summary``, ``status``, ``logs``.
            On failure: ``status``, ``error_message``, ``logs``.
        """
        logs: list[str] = list(state.get("logs", []))
        logger.info("SummarizationAgent.execute() started.")

        try:
            transcript = self._validate_transcript(state.get("transcript", ""))
            summary, elapsed = self._generate_summary(transcript)
            return self._update_state(summary, logs, elapsed)

        except Exception as exc:
            logger.error("SummarizationAgent failed: %s", exc, exc_info=True)
            logs.append(f"[SummarizationAgent] ERROR: {exc}")
            return {
                "status": ProcessingStatus.FAILED,
                "error_message": str(exc),
                "logs": logs,
            }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _validate_transcript(self, transcript: str) -> str:
        """
        Ensure the transcript is a non-empty string.

        Parameters
        ----------
        transcript : str
            Raw transcript from the state.

        Returns
        -------
        str
            Stripped transcript text.

        Raises
        ------
        ValueError
            If the transcript is missing or blank.
        """
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError(
                "Transcript is missing or empty. "
                "Ensure the Transcription Agent ran successfully."
            )
        logger.debug("Transcript validated — %d chars.", len(transcript))
        return transcript.strip()

    def _generate_summary(self, transcript: str) -> tuple[CallSummary, float]:
        """
        Delegate to OpenAIService and return a validated CallSummary with elapsed time.

        Parameters
        ----------
        transcript : str
            Validated transcript text.

        Returns
        -------
        tuple[CallSummary, float]
            Pydantic model populated by GPT-4o and elapsed seconds.
        """
        logger.info(
            "Summarization started — transcript length=%d chars", len(transcript)
        )
        start = time.perf_counter()
        summary = self._openai.generate_summary(transcript)
        elapsed = time.perf_counter() - start
        logger.info("Summary generated in %.2fs.", elapsed)
        return summary, elapsed

    def _update_state(
        self, summary: CallSummary, logs: list[str], elapsed: float
    ) -> dict[str, Any]:
        """
        Build the partial state update on successful summarisation.

        Parameters
        ----------
        summary : CallSummary
            Validated summary model.
        logs : list[str]
            Accumulated log entries.
        elapsed : float
            Time taken to generate the summary in seconds.

        Returns
        -------
        dict[str, Any]
            Partial state update for LangGraph to merge.
        """
        logs.append(
            f"[SummarizationAgent] Summary generated successfully — "
            f"customer_issue='{summary.customer_issue}', "
            f"sentiment={summary.customer_sentiment}, "
            f"action_items={len(summary.action_items)}, "
            f"key_topics={len(summary.key_topics)}, "
            f"elapsed={elapsed:.2f}s"
        )
        return {
            "summary": summary,
            "status": ProcessingStatus.PROCESSING,
            "error_message": None,
            "logs": logs,
        }
