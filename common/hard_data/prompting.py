from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = (
    "You are a medical AI assistant used for decision-making in a clinical setting."
)

USER_PROMPT_TEMPLATE = """Answer the following multiple-choice medical question.

Return only one compact JSON object in exactly this form: {{\"answer\":\"A\"}}
The answer must be one of A, B, C, or D. Do not provide reasoning, analysis, confidence, or any additional text.

Question:
{question}

Options:
A) {A}
B) {B}
C) {C}
D) {D}
"""

# This is appended after a complete user turn.  It bypasses Harmony's analysis
# channel and fixes the measurement at the first A-D answer token.
HARMONY_FINAL_ANSWER_PREFIX = (
    '<|start|>assistant<|channel|>final<|message|>{"answer":"'
)


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    options = row["options"]
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=row["question"],
        A=options["A"],
        B=options["B"],
        C=options["C"],
        D=options["D"],
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
