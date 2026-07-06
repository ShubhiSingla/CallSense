"""
services/whisper_service.py
---------------------------
Thin wrapper around the OpenAI Whisper API for audio transcription.

Uses the openai client's audio.transcriptions endpoint so no local
model download is required.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import openai

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".mp3", ".wav", ".m4a", ".flac", ".mp4"})

MIME_TYPES: dict[str, str] = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".mp4": "audio/mp4",
}


class WhisperService:
    """
    Transcribes audio files using the OpenAI Whisper API.

    Parameters
    ----------
    api_key : str | None
        OpenAI API key. Defaults to ``settings.OPENAI_API_KEY``.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = openai.OpenAI(api_key=api_key or settings.OPENAI_API_KEY)
        logger.debug("WhisperService initialised.")

    def transcribe(self, audio_path: str | Path) -> str:
        """
        Transcribe an audio file and return the plain transcript text.

        Parameters
        ----------
        audio_path : str | Path
            Path to the audio file.

        Returns
        -------
        str
            Full transcript text.

        Raises
        ------
        FileNotFoundError
            If the audio file does not exist.
        ValueError
            If the file extension is unsupported or the transcript is empty.
        openai.AuthenticationError
            If the API key is invalid.
        openai.APITimeoutError
            If the request times out.
        openai.APIConnectionError
            If a network error occurs.
        RuntimeError
            For any other unexpected API failure.
        """
        path = Path(audio_path).resolve()
        self._validate_audio(path)

        # Convert MP4 to MP3 before sending to Whisper
        converted_path: Path | None = None
        if path.suffix.lower() == ".mp4":
            path, converted_path = self._convert_mp4_to_mp3(path)

        logger.info("Whisper API started — file=%s", path.name)
        start = time.perf_counter()

        mime = MIME_TYPES.get(path.suffix.lower(), "audio/mpeg")

        try:
            with open(path, "rb") as audio_file:
                response = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=(path.name, audio_file, mime),
                    response_format="text",
                )
        except openai.AuthenticationError as exc:
            raise openai.AuthenticationError(
                "Invalid OpenAI API key. Check your .env file."
            ) from exc
        except openai.APITimeoutError as exc:
            raise openai.APITimeoutError(
                "Whisper API request timed out. Try again."
            ) from exc
        except openai.APIConnectionError as exc:
            raise openai.APIConnectionError(
                "Network error while contacting Whisper API."
            ) from exc
        except openai.APIError as exc:
            raise RuntimeError(f"Whisper API error: {exc}") from exc

        elapsed = time.perf_counter() - start
        transcript = response.strip() if isinstance(response, str) else response.text.strip()

        # Clean up the temporary converted file
        if converted_path and converted_path.exists():
            converted_path.unlink()

        if not transcript:
            raise ValueError("Whisper returned an empty transcript.")

        logger.info(
            "Whisper API completed — length=%d chars, elapsed=%.2fs",
            len(transcript),
            elapsed,
        )
        return transcript

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _convert_mp4_to_mp3(self, path: Path) -> tuple[Path, Path]:
        """
        Extract audio from an MP4 file and save it as MP3 using ffmpeg.

        Parameters
        ----------
        path : Path
            Path to the MP4 file.

        Returns
        -------
        tuple[Path, Path]
            (mp3_path, mp3_path) — converted file path and temp file to clean up.

        Raises
        ------
        RuntimeError
            If ffmpeg conversion fails.
        """
        tmp_dir = Path(tempfile.mkdtemp())
        mp3_path = tmp_dir / (path.stem + ".mp3")

        logger.info("Converting MP4 to MP3 via ffmpeg: %s -> %s", path.name, mp3_path.name)

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(path),
                "-vn",                  # drop video stream
                "-acodec", "libmp3lame",
                "-q:a", "2",
                str(mp3_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg conversion failed: {result.stderr[-300:]}"
            )

        logger.info("ffmpeg conversion complete: %s", mp3_path.name)
        return mp3_path, mp3_path

    def _validate_audio(self, path: Path) -> None:
        """
        Validate the audio file before sending to the API.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the extension is unsupported or the file is empty.
        """
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        if path.stat().st_size == 0:
            raise ValueError(f"Audio file is empty (0 bytes): {path}")
