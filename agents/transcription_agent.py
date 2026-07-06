"""
agents/transcription_agent.py
------------------------------
Transcription Agent — the second node in the CallSense-AI pipeline.

Reads audio_path from CallState, delegates to WhisperService,
and writes the transcript back into CallState.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from graph.state import CallState
from models.schemas import CallMetadata, ProcessingStatus
from services.whisper_service import WhisperService
from utils.logger import get_logger

logger = get_logger(__name__)


class TranscriptionAgent:
    """
    Converts a validated audio file into a text transcript via Whisper.

    Parameters
    ----------
    whisper_service : WhisperService | None
        Optional pre-built service instance. When ``None`` a new
        ``WhisperService`` is created using default settings.
    """

    def __init__(self, whisper_service: WhisperService | None = None) -> None:
        self._whisper = whisper_service or WhisperService()
        logger.debug("TranscriptionAgent initialised.")

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def execute(self, state: CallState) -> dict[str, Any]:
        """
        Transcribe the audio file referenced in *state*.

        Parameters
        ----------
        state : CallState
            Must contain ``audio_path``. Should also contain ``metadata``
            and ``logs`` from the Intake Agent.

        Returns
        -------
        dict[str, Any]
            Partial state update merged by LangGraph.
            On success: ``transcript``, ``metadata``, ``status``, ``logs``.
            On failure: ``status``, ``error_message``, ``logs``.
        """
        audio_path: str = state.get("audio_path", "")
        logs: list[str] = list(state.get("logs", []))

        logger.info("TranscriptionAgent.execute() — audio_path=%s", audio_path)

        try:
            validated_path = self._validate_audio(audio_path)
            transcript = self._transcribe_audio(validated_path)
            return self._update_state(state, transcript, logs)

        except Exception as exc:
            logger.error("TranscriptionAgent failed: %s", exc, exc_info=True)
            logs.append(f"[TranscriptionAgent] ERROR: {exc}")
            return {
                "status": ProcessingStatus.FAILED,
                "error_message": str(exc),
                "logs": logs,
            }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _validate_audio(self, audio_path: str) -> Path:
        """
        Confirm the audio file exists and is non-empty.

        Parameters
        ----------
        audio_path : str
            Raw path string from the state.

        Returns
        -------
        Path
            Resolved Path object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the path is blank or the file is empty.
        """
        if not audio_path:
            raise ValueError("audio_path is empty in CallState.")

        path = Path(audio_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if path.stat().st_size == 0:
            raise ValueError(f"Audio file is empty (0 bytes): {path}")

        logger.debug("Audio validated for transcription: %s", path)
        return path

    def _transcribe_audio(self, path: Path) -> str:
        """
        Delegate transcription to WhisperService and return the text.

        Parameters
        ----------
        path : Path
            Validated audio file path.

        Returns
        -------
        str
            Non-empty transcript text.

        Raises
        ------
        ValueError
            If Whisper returns an empty transcript.
        RuntimeError
            On any Whisper API failure.
        """
        logger.info("Whisper API started — file=%s", path.name)
        start = time.perf_counter()

        transcript = self._whisper.transcribe(str(path))

        elapsed = time.perf_counter() - start
        logger.info(
            "Whisper API completed — %d chars in %.2fs",
            len(transcript),
            elapsed,
        )
        return transcript

    def _update_state(
        self,
        state: CallState,
        transcript: str,
        logs: list[str],
    ) -> dict[str, Any]:
        """
        Build the partial state update on successful transcription.

        Updates ``metadata.duration_seconds`` and ``metadata.language``
        if a ``CallMetadata`` object is present in the state.

        Parameters
        ----------
        state : CallState
            Current state (read-only here).
        transcript : str
            Transcribed text.
        logs : list[str]
            Accumulated log entries.

        Returns
        -------
        dict[str, Any]
            Partial state update for LangGraph to merge.
        """
        word_count = len(transcript.split())
        logs.append(
            f"[TranscriptionAgent] OK — {word_count} words transcribed "
            f"from {Path(state.get('audio_path', '')).name}"
        )

        # Patch language onto existing CallMetadata if present
        existing: CallMetadata | None = state.get("metadata")
        if existing is not None:
            updated_metadata = existing.model_copy(update={"language": "en"})
        else:
            updated_metadata = existing

        logger.info(
            "Transcription complete — %d words, audio=%s",
            word_count,
            Path(state.get("audio_path", "")).name,
        )

        return {
            "transcript": transcript,
            "metadata": updated_metadata,
            "status": ProcessingStatus.PROCESSING,
            "error_message": None,
            "logs": logs,
        }
