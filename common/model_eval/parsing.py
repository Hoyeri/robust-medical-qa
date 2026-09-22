from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Iterable, Mapping

from .core import (
    CHOICES,
    RESPONSE_INSTRUCTION,
    ROLES,
    SYSTEM_PROMPT,
    VIEWS,
    canonical_sha256,
    model_messages,
    render_options,
    user_content,
    validate_source_pair,
)


CALIBRATION_SALT = "gpt-oss-only-hard-v2-external-eval-1024-calibration-v1"
NATIVE_FINAL_MARKERS = ("## Final Response", "<unused95>")
_FENCED_JSON = re.compile(
    r"\A```json\s*(\{.*\})\s*```\Z", flags=re.IGNORECASE | re.DOTALL
)
_ANSWER_FIELD = re.compile(
    r'["\']answer["\']\s*:\s*["\']([ABCD])["\']', flags=re.IGNORECASE
)
_FINAL_ANSWER_FIELD = re.compile(
    r'["\']answer["\']\s*:\s*["\']([ABCD])["\']\s*\}?\s*\Z',
    flags=re.IGNORECASE,
)


def prompt_contract() -> dict[str, Any]:
    return {
        "system_prompt": SYSTEM_PROMPT,
        "response_instruction": RESPONSE_INSTRUCTION,
        "required_field_order": ["rationale", "answer"],
        "answer_space": list(CHOICES),
        "view_order": list(VIEWS),
    }


def prompt_contract_sha256() -> str:
    return canonical_sha256(prompt_contract())


def response_compatibility_contract() -> dict[str, Any]:
    return {
        "primary_answer_modes": [
            "raw_json",
            "single_markdown_json_fence",
            "native_final_response_raw_json",
            "native_final_response_single_markdown_json_fence",
            "malformed_json_unique_final_answer_field",
        ],
        "native_final_markers": list(NATIVE_FINAL_MARKERS),
        "native_wrapper_policy": (
            "When a recognized model-native final-channel marker is present, only the "
            "suffix after its last occurrence is parsed. Earlier reasoning is never "
            "searched for an answer."
        ),
        "rationale_structured_modes": [
            "raw_json",
            "single_markdown_json_fence",
            "native_final_response_raw_json",
            "native_final_response_single_markdown_json_fence",
        ],
        "required_exact_fields": ["rationale", "answer"],
        "required_field_order": ["rationale", "answer"],
        "answer_space": list(CHOICES),
        "forbidden_recovery": [
            "inferring an answer from ordinary rationale prose",
            "searching model-native reasoning before the final-channel marker",
            "accepting multiple answer fields in the parsed final section",
            "accepting arbitrary text outside a permitted final JSON section",
        ],
    }


def response_compatibility_sha256() -> str:
    return canonical_sha256(response_compatibility_contract())


def _final_section(rendered: str) -> tuple[str, str]:
    located: list[tuple[int, str]] = []
    for marker in NATIVE_FINAL_MARKERS:
        index = rendered.rfind(marker)
        if index >= 0:
            located.append((index, marker))
    if not located:
        return rendered, ""
    index, marker = max(located)
    return rendered[index + len(marker) :].strip(), "native_final_response_"


def parse_compatible_response(text: str) -> dict[str, Any]:
    rendered = str(text).strip()
    candidate, prefix = _final_section(rendered)
    mode = f"{prefix}raw_json"
    fenced = _FENCED_JSON.fullmatch(candidate)
    if fenced is not None:
        candidate = fenced.group(1).strip()
        mode = f"{prefix}single_markdown_json_fence"
    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        fields = list(payload)
        rationale = payload.get("rationale")
        answer = str(payload.get("answer", "")).strip().upper()
        if fields == ["rationale", "answer"] and isinstance(rationale, str) and answer in CHOICES:
            return {
                "answer_valid": True,
                "rationale_structured_valid": True,
                "parse_mode": mode,
                "pred_answer": answer,
                "rationale": rationale,
                "field_order": fields,
            }
    final_answer = _FINAL_ANSWER_FIELD.search(candidate)
    json_shaped = candidate.startswith("{") and final_answer is not None
    matches = _ANSWER_FIELD.findall(candidate) if json_shaped else []
    if len(matches) == 1:
        return {
            "answer_valid": True,
            "rationale_structured_valid": False,
            "parse_mode": f"{prefix}malformed_json_unique_final_answer_field",
            "pred_answer": matches[0].upper(),
            "rationale": None,
            "field_order": None,
        }
    return {
        "answer_valid": False,
        "rationale_structured_valid": False,
        "parse_mode": "invalid",
        "pred_answer": None,
        "rationale": None,
        "field_order": None,
    }


def development_candidate_to_pair(row: Mapping[str, Any]) -> dict[str, Any]:
    selected_by_flip = bool(row.get("selected_by_flip_priority"))
    return {
        "question_id": f"calibration-{row['source_id']}",
        "source_idx": int(row["source_idx"]),
        "clean_question": str(row["clean_question"]),
        "distracted_question": str(row["question"]),
        "options": dict(row["options"]),
        "gold_answer": str(row["gold"]),
        "intended_target": str(row["intended_target"]),
        "added_distractor": str(row["distracting_sentence"]),
        "selection_pool": (
            "harmful_flip_priority" if selected_by_flip else "maximum_D_fallback"
        ),
        "construction_harmful_flip": selected_by_flip,
        "construction_candidate_top1_role": row.get("candidate_top1_role"),
    }
