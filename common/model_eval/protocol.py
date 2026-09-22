from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from .core import CHOICES
from .parsing import (
    canonical_sha256,
    model_messages,
    parse_compatible_response,
    prompt_contract,
    prompt_contract_sha256,
    response_compatibility_contract,
    response_compatibility_sha256,
)
from .calibration import CALIBRATION_SALT, select_calibration_pairs


FULL_VIEWS = ("clean", "meddistractqa", "hard")
CALIBRATION_VIEWS = ("clean", "hard")
STANDARD_ROLES = ("G", "T", "Other", "Invalid")
OFFICIAL_ROLES = ("G", "Wrong", "Invalid")
EXPECTED_PROTOCOL_STATUS = 'final_test_three_view_general_model_evaluation_v5_frozen_not_executed'
DIRECT_PROTOCOL_STATUS = 'medical_qa_direct_evaluation_v1'


def requires_calibration(protocol):
    status = protocol.get('status')
    if status not in (EXPECTED_PROTOCOL_STATUS, DIRECT_PROTOCOL_STATUS):
        raise ValueError('Unknown evaluation protocol')
    return status == EXPECTED_PROTOCOL_STATUS


def validate_source_triad(row: Mapping[str, Any]) -> None:
    required = {
        "question_id",
        "source_idx",
        "clean_question",
        "meddistractqa_question",
        "meddistractqa_distractor",
        "hard_question",
        "hard_distractor",
        "options",
        "gold_answer",
        "hard_intended_target",
        "hard_selection_pool",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"missing source triad fields: {missing}")
    if not isinstance(row["options"], Mapping) or set(row["options"]) != set(CHOICES):
        raise ValueError(f"invalid A-D options: {row.get('question_id')}")
    gold = str(row["gold_answer"])
    target = str(row["hard_intended_target"])
    if gold not in CHOICES or target not in CHOICES or gold == target:
        raise ValueError(f"invalid Gold/Hard-Target pair: {row.get('question_id')}")
    if row["hard_selection_pool"] not in {
        "harmful_flip_priority",
        "maximum_D_fallback",
    }:
        raise ValueError(f"invalid Hard selection pool: {row.get('question_id')}")
    official_sentence = str(row["meddistractqa_distractor"])
    hard_sentence = str(row["hard_distractor"])
    if not official_sentence or str(row["meddistractqa_question"]).count(official_sentence) != 1:
        raise ValueError(f"official distractor insertion mismatch: {row.get('question_id')}")
    if not hard_sentence or str(row["hard_question"]).count(hard_sentence) != 1:
        raise ValueError(f"Hard distractor insertion mismatch: {row.get('question_id')}")


def align_source_triads(
    hard_pairs: Iterable[Mapping[str, Any]],
    official_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    official = list(official_rows)
    official_index: dict[int, Mapping[str, Any]] = {}
    for row in official:
        source_idx = int(row["idx"])
        if source_idx in official_index:
            raise ValueError(f"duplicate official MedDistractQA source_idx: {source_idx}")
        official_index[source_idx] = row
    triads: list[dict[str, Any]] = []
    seen: set[int] = set()
    for hard in hard_pairs:
        source_idx = int(hard["source_idx"])
        if source_idx in seen:
            raise ValueError(f"duplicate Hard source_idx: {source_idx}")
        seen.add(source_idx)
        if source_idx not in official_index:
            raise ValueError(f"missing official MedDistractQA row: {source_idx}")
        baseline = official_index[source_idx]
        question_id = str(hard["question_id"])
        if str(baseline["source_id"]) != question_id:
            raise ValueError(f"source ID mismatch: {source_idx}")
        if str(baseline["clean_question"]) != str(hard["clean_question"]):
            raise ValueError(f"Clean question mismatch: {question_id}")
        if dict(baseline["options"]) != dict(hard["options"]):
            raise ValueError(f"option mismatch: {question_id}")
        if str(baseline["answer_idx"]) != str(hard["gold_answer"]):
            raise ValueError(f"Gold mismatch: {question_id}")
        if baseline.get("condition") not in {"B0_BS_OFFICIAL", "B0_NL_OFFICIAL"}:
            raise ValueError(f"not an official MedDistractQA Bystander/Nonliteral row: {question_id}")
        if baseline.get("b0_sampled_option") is not None:
            raise ValueError(f"official MedDistractQA unexpectedly has a target: {question_id}")
        triad = {
            "schema_version": "gpt_oss_only_hard_v2_final_test_three_view_triad_v5",
            "question_id": question_id,
            "source_idx": source_idx,
            "clean_question": str(hard["clean_question"]),
            "meddistractqa_question": str(baseline["question"]),
            "meddistractqa_distractor": str(baseline["distracting_sentence"]),
            "meddistractqa_condition": str(baseline["condition"]),
            "meddistractqa_target_available": False,
            "hard_question": str(hard["distracted_question"]),
            "hard_distractor": str(hard["added_distractor"]),
            "options": dict(hard["options"]),
            "gold_answer": str(hard["gold_answer"]),
            "hard_intended_target": str(hard["intended_target"]),
            "hard_selection_pool": str(hard["selection_pool"]),
            "hard_construction_harmful_flip": bool(
                hard.get("construction_harmful_flip", False)
            ),
            "hard_construction_candidate_top1_role": hard.get(
                "construction_candidate_top1_role"
            ),
            "must_not_be_used_for_training": True,
        }
        validate_source_triad(triad)
        triads.append(triad)
    return sorted(triads, key=lambda row: (int(row["source_idx"]), row["question_id"]))


def _request(
    *,
    request_set: str,
    question_id: str,
    source_idx: int,
    view: str,
    question: str,
    options: Mapping[str, Any],
    gold_answer: str,
    hard_intended_target: str,
    view_distractor: str | None,
    hard_selection_pool: str,
    hard_construction_harmful_flip: bool,
    hard_construction_candidate_top1_role: Any,
) -> dict[str, Any]:
    intended_target = hard_intended_target if view in {"clean", "hard"} else None
    return {
        "schema_version": (
            "gpt_oss_only_hard_v2_final_test_three_view_external_eval_request_v5"
        ),
        "request_id": (
            "gpt-oss-hard-v2-final-test-three-view-eval:"
            f"{request_set}:{question_id}:{view}:v5"
        ),
        "request_set": request_set,
        "question_id": question_id,
        "source_idx": source_idx,
        "view": view,
        "question": question,
        "options": dict(options),
        "gold_answer": gold_answer,
        "intended_target": intended_target,
        "hard_intended_target": hard_intended_target,
        "view_distractor": view_distractor,
        "hard_selection_pool": hard_selection_pool,
        "hard_construction_harmful_flip": hard_construction_harmful_flip,
        "hard_construction_candidate_top1_role": (
            hard_construction_candidate_top1_role
        ),
        "official_meddistractqa_target_available": False,
        "must_not_be_used_for_training": True,
    }


def build_calibration_requests(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in seen:
            raise ValueError(f"duplicate calibration question_id: {question_id}")
        seen.add(question_id)
        for view in CALIBRATION_VIEWS:
            requests.append(
                _request(
                    request_set="calibration",
                    question_id=question_id,
                    source_idx=int(row["source_idx"]),
                    view=view,
                    question=str(row[f"{view if view == 'clean' else 'distracted'}_question"]),
                    options=row["options"],
                    gold_answer=str(row["gold_answer"]),
                    hard_intended_target=str(row["intended_target"]),
                    view_distractor=(
                        None if view == "clean" else str(row["added_distractor"])
                    ),
                    hard_selection_pool=str(row["selection_pool"]),
                    hard_construction_harmful_flip=bool(
                        row.get("construction_harmful_flip", False)
                    ),
                    hard_construction_candidate_top1_role=row.get(
                        "construction_candidate_top1_role"
                    ),
                )
            )
    if len({row["request_id"] for row in requests}) != len(requests):
        raise AssertionError("calibration request IDs are not unique")
    return requests


def build_full_requests(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        validate_source_triad(row)
        question_id = str(row["question_id"])
        if question_id in seen:
            raise ValueError(f"duplicate full question_id: {question_id}")
        seen.add(question_id)
        question_by_view = {
            "clean": str(row["clean_question"]),
            "meddistractqa": str(row["meddistractqa_question"]),
            "hard": str(row["hard_question"]),
        }
        distractor_by_view = {
            "clean": None,
            "meddistractqa": str(row["meddistractqa_distractor"]),
            "hard": str(row["hard_distractor"]),
        }
        for view in FULL_VIEWS:
            requests.append(
                _request(
                    request_set="full",
                    question_id=question_id,
                    source_idx=int(row["source_idx"]),
                    view=view,
                    question=question_by_view[view],
                    options=row["options"],
                    gold_answer=str(row["gold_answer"]),
                    hard_intended_target=str(row["hard_intended_target"]),
                    view_distractor=distractor_by_view[view],
                    hard_selection_pool=str(row["hard_selection_pool"]),
                    hard_construction_harmful_flip=bool(
                        row.get("hard_construction_harmful_flip", False)
                    ),
                    hard_construction_candidate_top1_role=row.get(
                        "hard_construction_candidate_top1_role"
                    ),
                )
            )
    expected = len(seen) * len(FULL_VIEWS)
    if len(requests) != expected:
        raise AssertionError("three-view request factorial is incomplete")
    if len({row["request_id"] for row in requests}) != expected:
        raise AssertionError("full request IDs are not unique")
    return requests


def scientific_role_for_view(
    answer: str | None,
    *,
    view: str,
    gold: str,
    hard_target: str,
) -> str:
    if answer == gold:
        return "G"
    if answer not in CHOICES:
        return "Invalid"
    if view == "meddistractqa":
        return "Wrong"
    if answer == hard_target:
        return "T"
    return "Other"


def normalize_output(row: Mapping[str, Any]) -> dict[str, Any]:
    parsed = parse_compatible_response(str(row.get("generated_text", "")))
    answer = parsed["pred_answer"]
    view = str(row["view"])
    gold = str(row["gold_answer"])
    hard_target = str(row["hard_intended_target"])
    role = scientific_role_for_view(
        answer, view=view, gold=gold, hard_target=hard_target
    )
    return {
        **dict(row),
        **parsed,
        "scientific_role": role,
        "answer_correct": answer == gold,
        "target_selected": (
            answer == hard_target if view in {"clean", "hard"} else None
        ),
    }


def _view_counts(rows: list[dict[str, Any]], roles: tuple[str, ...]) -> dict[str, Any]:
    counts = Counter(str(row["scientific_role"]) for row in rows)
    denominator = len(rows)
    correct = counts.get("G", 0)
    return {
        "n": denominator,
        "correct": {"count": correct, "denominator": denominator},
        "accuracy": correct / denominator if denominator else None,
        "answer_valid": sum(bool(row["answer_valid"]) for row in rows),
        "rationale_structured_valid": sum(
            bool(row["rationale_structured_valid"]) for row in rows
        ),
        "max_token_cutoff": sum(bool(row.get("max_token_cutoff")) for row in rows),
        "role_counts": {role: counts.get(role, 0) for role in roles},
    }


def analyze_triad_outputs(
    rows: Iterable[Mapping[str, Any]], *, expected_questions: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = [normalize_output(row) for row in rows]
    expected_outputs = expected_questions * len(FULL_VIEWS)
    if len(normalized) != expected_outputs:
        raise ValueError(
            f"expected {expected_outputs} three-view outputs, observed {len(normalized)}"
        )
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in normalized:
        question_id = str(row["question_id"])
        view = str(row["view"])
        if view not in FULL_VIEWS:
            raise ValueError(f"unexpected full-evaluation view: {view}")
        if view in grouped.setdefault(question_id, {}):
            raise ValueError(f"duplicate output: {question_id}:{view}")
        grouped[question_id][view] = row
    if len(grouped) != expected_questions or any(
        set(triad) != set(FULL_VIEWS) for triad in grouped.values()
    ):
        raise ValueError("Clean/MedDistractQA/Hard grouping is incomplete")

    by_view = {
        "clean": _view_counts(
            [grouped[key]["clean"] for key in sorted(grouped)], STANDARD_ROLES
        ),
        "meddistractqa": _view_counts(
            [grouped[key]["meddistractqa"] for key in sorted(grouped)], OFFICIAL_ROLES
        ),
        "hard": _view_counts(
            [grouped[key]["hard"] for key in sorted(grouped)], STANDARD_ROLES
        ),
    }

    clean_to_official: Counter[str] = Counter()
    clean_to_hard: Counter[str] = Counter()
    clean_correct_ids: list[str] = []
    for question_id in sorted(grouped):
        clean_role = str(grouped[question_id]["clean"]["scientific_role"])
        official_role = str(
            grouped[question_id]["meddistractqa"]["scientific_role"]
        )
        hard_role = str(grouped[question_id]["hard"]["scientific_role"])
        clean_to_official[f"{clean_role}_to_{official_role}"] += 1
        clean_to_hard[f"{clean_role}_to_{hard_role}"] += 1
        if clean_role == "G":
            clean_correct_ids.append(question_id)

    denominator = len(clean_correct_ids)
    official_roles = Counter(
        str(grouped[key]["meddistractqa"]["scientific_role"])
        for key in clean_correct_ids
    )
    hard_roles = Counter(
        str(grouped[key]["hard"]["scientific_role"])
        for key in clean_correct_ids
    )
    paired_correctness = Counter(
        (
            "official_G" if grouped[key]["meddistractqa"]["scientific_role"] == "G" else "official_not_G",
            "hard_G" if grouped[key]["hard"]["scientific_role"] == "G" else "hard_not_G",
        )
        for key in clean_correct_ids
    )

    def rate(count: int) -> float | None:
        return count / denominator if denominator else None

    conditional = {
        "denominator": denominator,
        "meddistractqa": {
            "G_retained": official_roles.get("G", 0),
            "G_to_Wrong": official_roles.get("Wrong", 0),
            "G_to_Invalid": official_roles.get("Invalid", 0),
            "G_retained_rate": rate(official_roles.get("G", 0)),
            "G_to_Wrong_rate": rate(official_roles.get("Wrong", 0)),
            "G_to_Invalid_rate": rate(official_roles.get("Invalid", 0)),
            "intended_target_available": False,
        },
        "hard": {
            "G_retained": hard_roles.get("G", 0),
            "G_to_T": hard_roles.get("T", 0),
            "G_to_Other": hard_roles.get("Other", 0),
            "G_to_Invalid": hard_roles.get("Invalid", 0),
            "G_retained_rate": rate(hard_roles.get("G", 0)),
            "G_to_T_rate": rate(hard_roles.get("T", 0)),
            "G_to_Other_rate": rate(hard_roles.get("Other", 0)),
            "G_to_Invalid_rate": rate(hard_roles.get("Invalid", 0)),
        },
        "official_vs_hard_paired_correctness": {
            "both_G": paired_correctness.get(("official_G", "hard_G"), 0),
            "official_G_hard_not_G": paired_correctness.get(
                ("official_G", "hard_not_G"), 0
            ),
            "official_not_G_hard_G": paired_correctness.get(
                ("official_not_G", "hard_G"), 0
            ),
            "both_not_G": paired_correctness.get(
                ("official_not_G", "hard_not_G"), 0
            ),
        },
    }
    return {
        "questions": expected_questions,
        "outputs": len(normalized),
        "by_view": by_view,
        "own_clean_correct": conditional,
        "clean_to_meddistractqa_transitions": {
            f"{left}_to_{right}": clean_to_official.get(f"{left}_to_{right}", 0)
            for left in STANDARD_ROLES
            for right in OFFICIAL_ROLES
        },
        "clean_to_hard_transitions": {
            f"{left}_to_{right}": clean_to_hard.get(f"{left}_to_{right}", 0)
            for left in STANDARD_ROLES
            for right in STANDARD_ROLES
        },
    }, normalized


__all__ = [
    "CALIBRATION_SALT",
    "CALIBRATION_VIEWS",
    "FULL_VIEWS",
    "OFFICIAL_ROLES",
    "STANDARD_ROLES",
    "align_source_triads",
    "analyze_triad_outputs",
    "build_calibration_requests",
    "build_full_requests",
    "canonical_sha256",
    "model_messages",
    "normalize_output",
    "parse_compatible_response",
    "prompt_contract",
    "prompt_contract_sha256",
    "response_compatibility_contract",
    "response_compatibility_sha256",
    "scientific_role_for_view",
    "select_calibration_pairs",
    "validate_source_triad",
]


def analyze_pair_outputs(rows, *, expected_questions):
    normalized=[normalize_output(r) for r in rows]
    grouped={}
    for row in normalized:
        key,view=row['question_id'],row['view']
        if view not in ('clean','hard') or view in grouped.setdefault(key,{}):
            raise ValueError('Unexpected or duplicate evaluation view')
        grouped[key][view]=row
    if len(grouped)!=expected_questions or any(set(p)!=set(('clean','hard')) for p in grouped.values()):
        raise ValueError('Incomplete Clean/Hard pairs')
    clean_correct=[p for p in grouped.values() if p['clean']['scientific_role']=='G']
    n=len(clean_correct)
    roles=Counter(p['hard']['scientific_role'] for p in clean_correct)
    return dict(questions=expected_questions,outputs=len(normalized),
        by_view={v:_view_counts([p[v] for p in grouped.values()],STANDARD_ROLES) for v in ('clean','hard')},
        own_clean_correct=dict(denominator=n,hard={role:dict(count=roles[role],rate=roles[role]/n if n else None) for role in STANDARD_ROLES}),
        clean_to_hard_transitions=dict(Counter(p['clean']['scientific_role']+'_to_'+p['hard']['scientific_role'] for p in grouped.values()))),normalized
