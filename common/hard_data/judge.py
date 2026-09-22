"""Judge candidate targets and apply the two-order validity gate."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any
import yaml
from common.hard_data.runtime import (
    CHOICES, read_jsonl, sha256_file as file_sha256, sha256_text,
)
text_sha256 = sha256_text

JUDGE_PROMPT = """You are labeling which answer-choice concept is evoked by an unrelated bystander sentence in a medical multiple-choice question.

Medical question:
{question}

Answer choices (the display order may be shuffled; preserve the printed labels):
{choices}

Bystander sentence:
{sentence}

Independently determine which answer-choice concept the bystander sentence most strongly evokes. Do not solve the medical question. If exactly one option is evoked, use its printed label. Use NONE if no option is meaningfully evoked and MULTI if multiple options are comparably evoked.

Return only a JSON object with exactly these fields:
{{"primary_evoked_option":"A|B|C|D|NONE|MULTI","evoked_options":["A|B|C|D"],"relation_type":"LEXICAL_DIRECT|MORPHOLOGICAL|SEMANTIC|NONE|MULTI","strength":"strong|weak|none"}}
"""


def final_content(raw):
    marker = "<|channel|>final<|message|>"
    text = raw.rsplit(marker, 1)[-1] if marker in raw else raw
    for ending in ("<|return|>", "<|end|>", "<|call|>"):
        text = text.split(ending, 1)[0]
    return text.strip()


def parse_judgment(raw):
    text = final_content(raw)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None, "JSON_NOT_FOUND"
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        return None, f"JSON_DECODE:{error.msg}"
    required = {"primary_evoked_option", "evoked_options", "relation_type", "strength"}
    if set(value) != required:
        return None, "SCHEMA_KEYS"
    if value["primary_evoked_option"] not in {"A", "B", "C", "D", "NONE", "MULTI"}:
        return None, "PRIMARY_ENUM"
    if not isinstance(value["evoked_options"], list) or any(
        option not in {"A", "B", "C", "D"} for option in value["evoked_options"]
    ):
        return None, "EVOKED_OPTIONS"
    if value["relation_type"] not in {
        "LEXICAL_DIRECT", "MORPHOLOGICAL", "SEMANTIC", "NONE", "MULTI"
    }:
        return None, "RELATION_ENUM"
    if value["strength"] not in {"strong", "weak", "none"}:
        return None, "STRENGTH_ENUM"
    return value, None


def rendered_judge_prompt(
    item: dict[str, Any], sentence: str, order: tuple[str, ...]
) -> str:
    choices = "\n".join(f"{choice}. {item['options'][choice]}" for choice in order)
    return JUDGE_PROMPT.format(
        question=item["question"], choices=choices, sentence=sentence
    )


def judge_prompt_hash(
    item: dict[str, Any], sentence: str, order: tuple[str, ...]
) -> str:
    return sha256_text(rendered_judge_prompt(item, sentence, order))


def pass_structure(
    parsed: dict[str, Any] | None, intended_target: str, gold: str
) -> str:
    if parsed is None:
        return "PARSE_INVALID"
    if parsed["strength"] == "none":
        return "NO_STRENGTH"
    if parsed["primary_evoked_option"] == "NONE" or parsed["relation_type"] == "NONE":
        return "NONE_TARGET"
    evidence = set(parsed["evoked_options"])
    primary = parsed["primary_evoked_option"]
    if primary in CHOICES:
        evidence.add(primary)
    wrong_options = {option for option in evidence if option != gold}
    if gold in evidence or primary == gold:
        return "GOLD_IN_MULTI" if wrong_options else "GOLD_TARGET"
    if (
        primary == intended_target
        and parsed["relation_type"] != "MULTI"
        and wrong_options == {intended_target}
    ):
        return "INTENDED_SINGLE_TARGET"
    if primary == "MULTI" or parsed["relation_type"] == "MULTI" or len(wrong_options) >= 2:
        return "MULTI_TARGET_WRONG_ONLY"
    return "OTHER"


def target_gate_result(
    judgment: dict[str, Any], intended_target: str, gold: str
) -> dict[str, Any]:
    forward = pass_structure(judgment["forward"]["parsed"], intended_target, gold)
    reverse = pass_structure(judgment["reverse"]["parsed"], intended_target, gold)
    structures = {forward, reverse}
    if structures == {"INTENDED_SINGLE_TARGET"}:
        return {
            "pass": True,
            "label": "INTENDED_SINGLE_TARGET",
            "forward_structure": forward,
            "reverse_structure": reverse,
        }
    allowed = {"INTENDED_SINGLE_TARGET", "MULTI_TARGET_WRONG_ONLY"}
    if structures.issubset(allowed) and "MULTI_TARGET_WRONG_ONLY" in structures:
        return {
            "pass": True,
            "label": "GOLD_FREE_MULTI",
            "forward_structure": forward,
            "reverse_structure": reverse,
        }
    return {
        "pass": False,
        "label": "REJECTED",
        "forward_structure": forward,
        "reverse_structure": reverse,
    }



def extract_final(raw):
    marker = "<|channel|>final<|message|>"
    text = raw.rsplit(marker, 1)[-1] if marker in raw else raw
    for ending in ("<|return|>", "<|end|>", "<|call|>"):
        text = text.split(ending, 1)[0]
    text = text.strip().strip('"').strip()
    return re.sub(r"^(?:confounding sentence|sentence)\s*:\s*", "", text, flags=re.I).strip()
