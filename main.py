"""
main.py
-------
CallSense-AI entry point.

Launches the Streamlit UI. Run with:
    python main.py

Or directly with Streamlit:
    streamlit run ui/streamlit_app.py
"""

import subprocess
import sys


def main() -> None:
    """Launch the Streamlit application."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py"],
        check=True,
    )


if __name__ == "__main__":
    main()
