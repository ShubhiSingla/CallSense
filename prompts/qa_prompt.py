"""
prompts/qa_prompt.py
--------------------
Prompt templates used by the QualityScoreAgent.

Scoring criteria are embedded in the system prompt so the LLM
evaluates every call against the same rubric consistently.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ------------------------------------------------------------------ #
# System instruction
# ------------------------------------------------------------------ #

QA_SYSTEM_TEMPLATE: str = """
You are a senior call centre quality assurance evaluator.
Your task is to score a customer support call based on the transcript
and summary provided.

Scoring Rubric (each dimension is scored 0.0 – 10.0):

1. empathy_score
   - Did the agent acknowledge the customer's feelings?
   - Did the agent use empathetic language?

2. resolution_score
   - Was the customer's issue fully resolved?
   - If not, was a clear next step provided?

3. communication_score
   - Was the agent clear, professional, and concise?
   - Were there unnecessary filler words or confusion?

4. compliance_passed (boolean)
   - Did the agent use required opening/closing phrases?
   - Were data protection / privacy statements made where required?

Compute overall_score as the average of the three numeric dimensions.

Return your response as valid JSON matching the schema below.

JSON Schema:
{{
  "overall_score": <float 0.0–10.0>,
  "empathy_score": <float 0.0–10.0>,
  "resolution_score": <float 0.0–10.0>,
  "communication_score": <float 0.0–10.0>,
  "compliance_passed": <true | false>,
  "feedback": "<narrative feedback for the agent>"
}}
""".strip()

# ------------------------------------------------------------------ #
# Human / user message
# ------------------------------------------------------------------ #

QA_HUMAN_TEMPLATE: str = """
Transcript:
---
{transcript}
---

Summary:
---
{summary}
---

Please evaluate this call and return the QA score JSON.
""".strip()

# ------------------------------------------------------------------ #
# Assembled ChatPromptTemplate
# ------------------------------------------------------------------ #

QA_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(QA_SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(QA_HUMAN_TEMPLATE),
    ]
)

# TODO: Introduce per-client rubric overrides loaded from config.
# TODO: Add a calibration prompt that aligns scores with human raters.
