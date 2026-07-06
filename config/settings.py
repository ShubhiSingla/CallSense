"""
config/settings.py
------------------
Loads environment variables and exposes typed configuration.
Import the `settings` singleton anywhere in the project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level above config/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Settings:
    """Application-wide settings backed by environment variables."""

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "CallSenseAI")
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

    APP_NAME: str = "CallSense-AI"
    APP_VERSION: str = "1.0"

    def validate(self) -> None:
        """Raise ValueError if any required setting is missing."""
        if not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )


# Singleton — import this everywhere
settings = Settings()
