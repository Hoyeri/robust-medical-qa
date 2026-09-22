#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.model_eval.protocol import (  # noqa: E402
    EXPECTED_PROTOCOL_STATUS,
    requires_calibration,
    FULL_VIEWS,
    analyze_triad_outputs,
    canonical_sha256,
    normalize_output,
    prompt_contract_sha256,
    response_compatibility_sha256,
)
from common.hard_data.runtime import (  # noqa: E402
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


TRACK_KEY = "general_models_three_view_rationale_1024"


def quantile(values: list[int], probability: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def token_length_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [int(row["generated_token_count"]) for row in rows]
    return {
        "n": len(values),
        "minimum": min(values) if values else None,
        "median": quantile(values, 0.5),
        "p90": quantile(values, 0.9),
        "p95": quantile(values, 0.95),
        "maximum": max(values) if values else None,
        "mean": (sum(values) / len(values)) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--mode", choices=("calibration", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = read_json(args.protocol)
    calibrated = requires_calibration(protocol)
    if args.mode == 'calibration' and not calibrated:
        raise ValueError('Direct evaluation has no calibration stage')
    if protocol["evaluation"]["track_key"] != TRACK_KEY:
        raise RuntimeError("final-test V5 track identity drift")
    if args.model_key not in protocol["primary_model_keys"]:
        raise ValueError("model is not in the frozen V5 primary pool")
    max_new_tokens = int(protocol["evaluation"]["max_new_tokens"])

    run_manifest = read_json(args.run_manifest)
    if run_manifest.get("status") != "complete":
        raise RuntimeError("generation has unresolved technical failures")
    if (
        run_manifest.get("model_key") != args.model_key
        or run_manifest.get("track_key") != TRACK_KEY
        or run_manifest.get("mode") != args.mode
    ):
        raise RuntimeError("generation manifest identity mismatch")
    protocol_hash = sha256_file(args.protocol)
    if protocol_hash != run_manifest["protocol_sha256"]:
        raise RuntimeError("generation used a different final-test V5 protocol")
    if int(run_manifest.get("max_new_tokens", -1)) != max_new_tokens:
        raise RuntimeError("generation did not use the frozen 1024-token budget")
    if sha256_file(args.requests) != run_manifest["bindings"]["requests"]["sha256"]:
        raise RuntimeError("generation used different requests")
    if sha256_file(args.outputs) != run_manifest["bindings"]["outputs"]["sha256"]:
        raise RuntimeError("generation output hash mismatch")
    if protocol["prompt_contract"]["sha256"] != prompt_contract_sha256():
        raise RuntimeError("final-test V5 prompt contract drift")
    if (
        protocol["response_compatibility_contract"]["sha256"]
        != response_compatibility_sha256()
    ):
        raise RuntimeError("final-test V5 response parser contract drift")

    requests = read_jsonl(args.requests)
    rows = read_jsonl(args.outputs)
    expected_ids = {str(row["request_id"]) for row in requests}
    observed_ids = {str(row["request_id"]) for row in rows}
    problems: list[str] = []
    if len(expected_ids) != len(requests):
        problems.append("duplicate_request_id")
    if len(observed_ids) != len(rows):
        problems.append("duplicate_output_id")
    if expected_ids != observed_ids:
        problems.append("request_output_coverage_mismatch")
    request_index = {str(row["request_id"]): row for row in requests}
    for row in rows:
        request_id = str(row["request_id"])
        if row.get("status") != "ok":
            problems.append(f"technical_status:{request_id}")
        if row.get("model_key") != args.model_key:
            problems.append(f"model_key:{request_id}")
        if row.get("track_key") != TRACK_KEY:
            problems.append(f"track_key:{request_id}")
        if int(row.get("max_new_tokens", -1)) != max_new_tokens:
            problems.append(f"budget:{request_id}")
        if row.get("protocol_sha256") != protocol_hash:
            problems.append(f"protocol:{request_id}")
        source = request_index.get(request_id)
        if source is None or row.get("request_sha256") != canonical_sha256(source):
            problems.append(f"request_binding:{request_id}")

    normalized = [normalize_output(row) for row in rows]
    technical_failures = sum(row.get("status") != "ok" for row in normalized)
    answer_valid = sum(bool(row.get("answer_valid")) for row in normalized)
    rationale_valid = sum(
        bool(row.get("rationale_structured_valid")) for row in normalized
    )
    cutoff = sum(bool(row.get("max_token_cutoff")) for row in normalized)
    observed_views = sorted({str(row["view"]) for row in normalized})
    token_usage = {
        "all": token_length_summary(normalized),
        **{
            view: token_length_summary(
                [row for row in normalized if row.get("view") == view]
            )
            for view in observed_views
        },
    }
    audit = {
        "schema_version": (
            "gpt_oss_only_hard_v2_final_test_three_view_technical_audit_v5"
        ),
        "status": "complete" if not problems else "failed",
        "track_key": TRACK_KEY,
        "model_key": args.model_key,
        "mode": args.mode,
        "views": observed_views,
        "max_new_tokens": max_new_tokens,
        "counts": {
            "expected": len(requests),
            "observed": len(rows),
            "problems": len(problems),
            "technical_failures": technical_failures,
            "answer_valid": answer_valid,
            "rationale_structured_valid": rationale_valid,
            "max_token_cutoff": cutoff,
        },
        "generated_token_count": token_usage,
        "problems": problems[:100],
        "bindings": {
            "protocol": {"path": str(args.protocol), "sha256": protocol_hash},
            "requests": {"path": str(args.requests), "sha256": sha256_file(args.requests)},
            "outputs": {"path": str(args.outputs), "sha256": sha256_file(args.outputs)},
            "run_manifest": {"path": str(args.run_manifest), "sha256": sha256_file(args.run_manifest)},
        },
        "must_not_be_used_for_training": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "technical_audit.json"
    write_json(audit_path, audit, overwrite=True)
    if problems:
        raise RuntimeError(f"technical audit failed with {len(problems)} problems")

    normalized_path = args.output_dir / "normalized_outputs.jsonl"
    write_jsonl(normalized_path, normalized, overwrite=True)

    if args.mode == "calibration":
        gate_contract = protocol["calibration_gate"]
        denominator = len(normalized)
        answer_valid_rate = answer_valid / denominator
        cutoff_rate = cutoff / denominator
        checks = {
            "minimum_answer_valid_rate": {
                "threshold": float(gate_contract["minimum_answer_valid_rate"]),
                "observed": answer_valid_rate,
                "passed": answer_valid_rate
                >= float(gate_contract["minimum_answer_valid_rate"]),
            },
            "maximum_max_token_cutoff_rate": {
                "threshold": float(gate_contract["maximum_max_token_cutoff_rate"]),
                "observed": cutoff_rate,
                "passed": cutoff_rate
                <= float(gate_contract["maximum_max_token_cutoff_rate"]),
            },
            "maximum_technical_failures": {
                "threshold": int(gate_contract["maximum_technical_failures"]),
                "observed": technical_failures,
                "passed": technical_failures
                <= int(gate_contract["maximum_technical_failures"]),
            },
        }
        passed = all(item["passed"] for item in checks.values())
        gate = {
            "schema_version": (
                "gpt_oss_only_hard_v2_final_test_three_view_calibration_gate_v5"
            ),
            "status": (
                "passed_final_test_three_view_v5_calibration_gate"
                if passed
                else "failed_final_test_three_view_v5_calibration_gate"
            ),
            "track_key": TRACK_KEY,
            "model_key": args.model_key,
            "protocol_sha256": protocol_hash,
            "max_new_tokens": max_new_tokens,
            "counts": {
                "questions": int(protocol["counts"]["calibration_pairs"]),
                "responses": denominator,
                "answer_valid": answer_valid,
                "max_token_cutoff": cutoff,
                "technical_failures": technical_failures,
                "rationale_structured_valid": rationale_valid,
            },
            "generated_token_count": token_usage,
            "checks": checks,
            "accuracy_inspected_or_used": False,
            "full_run_allowed": passed,
            "bindings": {
                "technical_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
                "normalized_outputs": {"path": str(normalized_path), "sha256": sha256_file(normalized_path)},
            },
            "must_not_be_used_for_training": True,
        }
        write_json(args.output_dir / "calibration_gate.json", gate, overwrite=True)
        print("Calibration passed" if passed else "Calibration failed")
        return

    if tuple(observed_views) not in (tuple(sorted(FULL_VIEWS)), ('clean','hard')):
        raise RuntimeError("full V5 output does not contain all three views")
    expected_questions = int(protocol["counts"]["full_triads"])
    from .protocol import analyze_pair_outputs
    analysis_function = analyze_triad_outputs if len(observed_views)==3 else analyze_pair_outputs
    summary, normalized = analysis_function(
        rows, expected_questions=expected_questions
    )
    write_jsonl(normalized_path, normalized, overwrite=True)
    analysis = {
        "schema_version": (
            "gpt_oss_only_hard_v2_final_test_three_view_model_analysis_v5"
        ),
        "status": "complete_before_cross_model_analysis",
        "track_key": TRACK_KEY,
        "model_key": args.model_key,
        "model": protocol["models"][args.model_key],
        "mode": "full",
        "max_new_tokens": max_new_tokens,
        "counts": summary,
        "generated_token_count": token_usage,
        "interpretation": {
            "unconditional_denominator_per_view": expected_questions,
            "source_population": int(protocol["counts"]["source_items"]),
            "construction_infeasible_exclusions": int(
                protocol["counts"]["construction_infeasible_exclusions"]
            ),
            "conditional_denominator": "this_model_clean_correct_only",
            "official_meddistractqa_target_available": False,
            "official_meddistractqa_G_to_Target_forbidden": True,
            "invalid_and_cutoff_are_not_technically_retried": True,
            "not_a_pure_scaling_or_specialization_effect": True,
        },
        "bindings": {
            "protocol": {"path": str(args.protocol), "sha256": protocol_hash},
            "outputs": {"path": str(args.outputs), "sha256": sha256_file(args.outputs)},
            "technical_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
            "normalized_outputs": {"path": str(normalized_path), "sha256": sha256_file(normalized_path)},
        },
        "must_not_be_used_for_training": True,
    }
    write_json(args.output_dir / "analysis.json", analysis, overwrite=True)
    print(f"Evaluation summarized: {expected_questions} questions")


if __name__ == "__main__":
    main()
