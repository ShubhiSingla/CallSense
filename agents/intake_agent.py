"""
agents/intake_agent.py
----------------------
Call Intake Agent — the first node in the CallSense-AI pipeline.

Responsibilities
----------------
- Verify the audio file exists.
- Verify it is a supported format (.mp3, .wav, .m4a, .flac).
- Validate file size (must be > 0 bytes).
- Extract file metadata (name, type, size, created time).
- Populate CallMetadata and update CallState.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from graph.state import CallState
from models.schemas import CallMetadata, ProcessingStatus
from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".mp3", ".wav", ".m4a", ".flac", ".mp4"})


class CallIntakeAgent:
    """Validates an incoming audio file and populates the initial CallState."""

    def __init__(self) -> None:
        logger.debug("CallIntakeAgent initialised.")

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def execute(self, state: CallState) -> dict[str, Any]:
        """
        Run intake validation and metadata extraction.

        Parameters
        ----------
        state : CallState
            Must contain ``audio_path``.

        Returns
        -------
        dict[str, Any]
            Partial state update merged by LangGraph.
            On success: ``metadata``, ``status``, ``logs``.
            On failure: ``status``, ``error_message``, ``logs``.
        """
        audio_path: str = state.get("audio_path", "")
        logs: list[str] = list(state.get("logs", []))
        logger.info("CallIntakeAgent.execute() — audio_path=%s", audio_path)

        try:
            validated_path = self._validate_file(audio_path)
            metadata = self._extract_metadata(validated_path)
            return self._update_state(metadata, logs)

        except (FileNotFoundError, ValueError) as exc:
            logger.error("Intake failed: %s", exc)
            logs.append(f"[IntakeAgent] ERROR: {exc}")
            return {
                "status": ProcessingStatus.FAILED,
                "error_message": str(exc),
                "logs": logs,
            }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _validate_file(self, audio_path: str) -> Path:
        """
        Validate the audio file path.

        Parameters
        ----------
        audio_path : str
            Raw path from the state.

        Returns
        -------
        Path
            Resolved Path if all checks pass.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the extension is unsupported or the file is empty.
        """
        if not audio_path:
            raise ValueError("audio_path is empty. Provide a valid file path.")

        path = Path(audio_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        if path.stat().st_size == 0:
            raise ValueError(f"Audio file is empty (0 bytes): {path}")

        logger.debug("File validated: %s", path)
        return path

    def _extract_metadata(self, path: Path) -> CallMetadata:
        """
        Extract file metadata without performing any AI inference.

        Parameters
        ----------
        path : Path
            Validated audio file path.

        Returns
        -------
        CallMetadata
            Populated metadata model.
        """
        stat = path.stat()

        # Use file creation time where available, otherwise fall back to mtime
        created_ts = stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_mtime
        uploaded_at = datetime.utcfromtimestamp(created_ts)

        metadata = CallMetadata(
            file_name=path.name,
            file_type=path.suffix.lower().lstrip("."),
            file_size_bytes=stat.st_size,
            duration_seconds=0.0,   # Populated by TranscriptionAgent (Whisper)
            language=None,          # Detected by TranscriptionAgent
            uploaded_at=uploaded_at,
        )

        logger.debug(
            "Metadata extracted — file=%s, size=%d bytes",
            metadata.file_name,
            metadata.file_size_bytes,
        )
        return metadata

    def _update_state(
        self, metadata: CallMetadata, logs: list[str]
    ) -> dict[str, Any]:
        """
        Build the partial state update dict on successful intake.

        Parameters
        ----------
        metadata : CallMetadata
            Extracted file metadata.
        logs : list[str]
            Existing log entries from the state.

        Returns
        -------
        dict[str, Any]
            Partial state update for LangGraph to merge.
        """
        logs.append(
            f"[IntakeAgent] OK — file={metadata.file_name}, "
            f"size={metadata.file_size_bytes} bytes"
        )
        return {
            "metadata": metadata,
            "status": ProcessingStatus.PROCESSING,
            "error_message": None,
            "logs": logs,
        }
