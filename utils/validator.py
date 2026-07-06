"""
utils/validator.py
------------------
Input validation helpers for CallSense-AI.

These utilities are used by agents and services to validate data
before processing begins, providing clear error messages early
rather than cryptic failures deep in the pipeline.
"""

from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

# Supported audio MIME / file extensions
SUPPORTED_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
)

# Minimum / maximum audio file size in bytes (1 KB – 500 MB)
MIN_AUDIO_BYTES: int = 1_024
MAX_AUDIO_BYTES: int = 500 * 1_024 * 1_024


class ValidationError(ValueError):
    """Raised when a validation check fails inside CallSense-AI."""


def validate_audio_file(path: str | Path) -> Path:
    """
    Validate that *path* points to a readable, supported audio file.

    Parameters
    ----------
    path:
        Filesystem path to the audio file.

    Returns
    -------
    Path
        Resolved ``Path`` object if all checks pass.

    Raises
    ------
    ValidationError
        If the file does not exist, has an unsupported extension,
        or falls outside the allowed size range.
    """
    resolved = Path(path).resolve()

    if not resolved.exists():
        raise ValidationError(f"Audio file not found: {resolved}")

    if not resolved.is_file():
        raise ValidationError(f"Path is not a file: {resolved}")

    if resolved.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValidationError(
            f"Unsupported audio format '{resolved.suffix}'. "
            f"Supported: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}"
        )

    size = resolved.stat().st_size
    if size < MIN_AUDIO_BYTES:
        raise ValidationError(
            f"Audio file is too small ({size} bytes). "
            f"Minimum: {MIN_AUDIO_BYTES} bytes."
        )
    if size > MAX_AUDIO_BYTES:
        raise ValidationError(
            f"Audio file is too large ({size} bytes). "
            f"Maximum: {MAX_AUDIO_BYTES} bytes."
        )

    logger.debug("Audio file validated: %s (%d bytes)", resolved, size)
    return resolved


def validate_transcript(text: Any) -> str:
    """
    Validate that *text* is a non-empty string suitable for downstream agents.

    Parameters
    ----------
    text:
        Value to validate.

    Returns
    -------
    str
        Stripped transcript string.

    Raises
    ------
    ValidationError
        If *text* is not a string or is blank.
    """
    if not isinstance(text, str):
        raise ValidationError(
            f"Transcript must be a string, got {type(text).__name__}."
        )
    stripped = text.strip()
    if not stripped:
        raise ValidationError("Transcript is empty.")
    logger.debug("Transcript validated (%d characters).", len(stripped))
    return stripped


def validate_qa_score(score: Any) -> float:
    """
    Validate that *score* is a float in the range [0.0, 10.0].

    Parameters
    ----------
    score:
        Value to validate.

    Returns
    -------
    float

    Raises
    ------
    ValidationError
        If *score* cannot be converted to float or is out of range.
    """
    try:
        value = float(score)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"QA score must be numeric, got {score!r}."
        ) from exc

    if not (0.0 <= value <= 10.0):
        raise ValidationError(
            f"QA score {value} is out of range [0.0, 10.0]."
        )
    return value
