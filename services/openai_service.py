"""
services/openai_service.py
--------------------------
GPT-4o wrapper using LangChain structured output.

OpenAIService is the single point of contact for all GPT-4o calls.
Uses with_structured_output() so responses are validated Pydantic
models — no manual JSON parsing required.
"""

from __future__ import annotations

import time

import openai
from langchain_openai import ChatOpenAI

from config.settings import settings
from models.schemas import CallSummary
from prompts.summary_prompt import SUMMARY_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIService:
    """
    Wrapper for GPT-4o interactions via LangChain.

    Parameters
    ----------
    model : str
        OpenAI model identifier. Defaults to ``settings.OPENAI_MODEL``.
    """

    def __init__(self, model: str = settings.OPENAI_MODEL) -> None:
        self._llm = ChatOpenAI(
            model=model,
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
        )
        logger.debug("OpenAIService initialised (model=%s).", model)

    def generate_summary(self, transcript: str) -> CallSummary:
        """
        Generate a structured call summary from a transcript.

        Uses LangChain's with_structured_output() to return a validated
        CallSummary Pydantic model directly — no JSON parsing needed.

        Parameters
        ----------
        transcript : str
            Raw transcript text from the TranscriptionAgent.

        Returns
        -------
        CallSummary
            Validated Pydantic model populated by GPT-4o.

        Raises
        ------
        ValueError
            If the transcript is empty.
        openai.AuthenticationError
            If the API key is invalid.
        openai.APITimeoutError
            If the request times out.
        openai.APIConnectionError
            If a network error occurs.
        RuntimeError
            For any other unexpected API failure.
        """
        if not transcript or not transcript.strip():
            raise ValueError("Transcript is empty — cannot generate summary.")

        logger.info("GPT request started — transcript length=%d chars", len(transcript))
        start = time.perf_counter()

        structured_llm = self._llm.with_structured_output(CallSummary)
        chain = SUMMARY_PROMPT | structured_llm

        try:
            summary: CallSummary = chain.invoke({"transcript": transcript})
        except openai.AuthenticationError as exc:
            raise openai.AuthenticationError(
                "Invalid OpenAI API key. Check your .env file."
            ) from exc
        except openai.APITimeoutError as exc:
            raise openai.APITimeoutError(
                "GPT-4o request timed out. Try again."
            ) from exc
        except openai.APIConnectionError as exc:
            raise openai.APIConnectionError(
                "Network error while contacting OpenAI API."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        elapsed = time.perf_counter() - start
        logger.info(
            "GPT response received — sentiment=%s, elapsed=%.2fs",
            summary.customer_sentiment,
            elapsed,
        )
        return summary
