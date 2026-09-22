from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Iterable, Mapping


CHOICES = ("A", "B", "C", "D")
VIEWS = ("clean", "distracted")
ROLES = ("G", "T", "Other", "Invalid")
SYSTEM_PROMPT = (
    "You are a medical AI assistant used for decision-making in a clinical setting."
)
RESPONSE_INSTRUCTION = (
    "Answer the following medical multiple-choice question.\n\n"
    'Return one JSON object with exactly two fields in this order: "rationale" and "answer".\n'
    "The answer must be one of A, B, C, or D."
)

_FENCED_JSON = re.compile(
    r"\A```json\s*(\{.*\})\s*```\Z", flags=re.IGNORECASE | re.DOTALL
)
_ANSWER_FIELD = re.compile(
    r'["\']answer["\']\s*:\s*["\']([ABCD])["\']', flags=re.IGNORECASE
)
_FINAL_ANSWER_FIELD = re.compile(
    r'["\']answer["\']\s*:\s*["\']([ABCD])["\']\s*\Z',
    flags=re.IGNORECASE,
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()










def render_options(options: Mapping[str, Any]) -> str:
    if set(options) != set(CHOICES):
        raise ValueError("options must contain exactly A-D")
    return "\n".join(f"{choice}) {str(options[choice]).strip()}" for choice in CHOICES)


def user_content(question: str, options: Mapping[str, Any]) -> str:
    return (
        f"{RESPONSE_INSTRUCTION}\n\n"
        f"Question:\n{str(question).strip()}\n\n"
        f"Options:\n{render_options(options)}"
    )


def model_messages(request: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_content(str(request["question"]), request["options"]),
        },
    ]


def validate_source_pair(row: Mapping[str, Any]) -> None:
    required = {
        "question_id",
        "source_idx",
        "clean_question",
        "distracted_question",
        "options",
        "gold_answer",
        "intended_target",
        "added_distractor",
        "selection_pool",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"missing source pair fields: {missing}")
    if not isinstance(row["options"], Mapping) or set(row["options"]) != set(CHOICES):
        raise ValueError(f"invalid A-D options: {row.get('question_id')}")
    gold = str(row["gold_answer"])
    target = str(row["intended_target"])
    if gold not in CHOICES or target not in CHOICES or gold == target:
        raise ValueError(f"invalid Gold/Target pair: {row.get('question_id')}")
    if row["selection_pool"] not in {"harmful_flip_priority", "maximum_D_fallback"}:
        raise ValueError(f"invalid construction selection pool: {row.get('question_id')}")
