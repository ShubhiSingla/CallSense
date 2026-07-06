"""
test_openai.py
--------------
Verifies that the OpenAI API key is configured correctly and
that GPT-4o is reachable. Run with:

    python test_openai.py
"""

from config.settings import settings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def main() -> None:
    """Send a test prompt to GPT-4o and print the response."""
    try:
        settings.validate()
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )

    try:
        response = llm.invoke([HumanMessage(content="Say Hello from CallSense AI.")])
        print(f"[SUCCESS] {response.content}")
    except Exception as exc:
        print(f"[ERROR] OpenAI call failed: {exc}")


if __name__ == "__main__":
    main()
