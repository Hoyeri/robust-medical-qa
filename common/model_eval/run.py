#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.model_eval.protocol import (  # noqa: E402
    EXPECTED_PROTOCOL_STATUS,
    requires_calibration,
    canonical_sha256,
    model_messages,
    parse_compatible_response,
    prompt_contract_sha256,
    response_compatibility_sha256,
    scientific_role_for_view,
)
from common.hard_data.runtime import (  # noqa: E402
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


TRACK_KEY = "general_models_three_view_rationale_1024"


def shard_path(root: Path, request_id: str) -> Path:
    return root / f"{hashlib.sha256(request_id.encode('utf-8')).hexdigest()}.json"


def accepts(signature: inspect.Signature, name: str) -> bool:
    return name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def require_passed_calibration(
    *, gate_path: Path, protocol_path: Path, model_key: str
) -> None:
    if not gate_path.is_file():
        raise FileNotFoundError(
            f"{model_key}: missing V5 calibration gate; run calibration first"
        )
    gate = read_json(gate_path)
    if gate.get("status") != "passed_final_test_three_view_v5_calibration_gate":
        raise RuntimeError(f"{model_key}: final-test V5 calibration did not pass")
    if gate.get("model_key") != model_key or gate.get("track_key") != TRACK_KEY:
        raise RuntimeError(f"{model_key}: V5 calibration gate identity mismatch")
    if gate.get("protocol_sha256") != sha256_file(protocol_path):
        raise RuntimeError(f"{model_key}: calibration used a different V5 protocol")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--mode", choices=("calibration", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-gate", type=Path)
    args = parser.parse_args()

    protocol = read_json(args.protocol)
    calibrated = requires_calibration(protocol)
    if args.mode == 'calibration' and not calibrated:
        raise ValueError('Direct evaluation has no calibration stage')
    if protocol["evaluation"]["track_key"] != TRACK_KEY:
        raise RuntimeError("final-test V5 track identity drift")
    if protocol["prompt_contract"]["sha256"] != prompt_contract_sha256():
        raise RuntimeError("final-test V5 prompt contract implementation drift")
    if (
        protocol["response_compatibility_contract"]["sha256"]
        != response_compatibility_sha256()
    ):
        raise RuntimeError("final-test V5 response parser implementation drift")
    if args.model_key not in protocol["primary_model_keys"]:
        raise ValueError(f"unknown or excluded V5 primary model: {args.model_key}")
    model = protocol["models"][args.model_key]
    if model["backend"] != "vllm_chat":
        raise RuntimeError("only the frozen vllm_chat backend is supported")
    max_new_tokens = int(protocol["evaluation"]["max_new_tokens"])

    if args.mode == "full" and calibrated:
        if args.calibration_gate is None:
            raise ValueError("--calibration-gate is required for a full V5 run")
        require_passed_calibration(
            gate_path=args.calibration_gate,
            protocol_path=args.protocol,
            model_key=args.model_key,
        )

    request_binding = protocol["bindings"][f"{args.mode}_requests"]
    if sha256_file(args.requests) != request_binding["sha256"]:
        raise RuntimeError("final-test V5 request file differs from frozen protocol")
    requests = read_jsonl(args.requests)
    expected = int(protocol["counts"][f"{args.mode}_requests_per_model"])
    if len(requests) != expected:
        raise RuntimeError(f"expected {expected} requests, observed {len(requests)}")

    try:
        observed_vllm = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("vLLM is not installed in the evaluation environment") from error
    expected_vllm = str(protocol["evaluation"]["vllm_version"])
    if observed_vllm != expected_vllm:
        raise RuntimeError(
            f"vLLM version mismatch: expected={expected_vllm} observed={observed_vllm}"
        )

    visible = [
        item
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item
    ]
    tensor_parallel_size = int(model["tensor_parallel_size"])
    if len(visible) != tensor_parallel_size:
        raise RuntimeError(
            f"{args.model_key} requires {tensor_parallel_size} visible GPU(s); "
            f"observed {visible}"
        )

    output_dir = args.output_dir.resolve()
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha256_file(args.protocol)
    completed: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for request in requests:
        request_id = str(request["request_id"])
        path = shard_path(shards_dir, request_id)
        if path.is_file():
            existing = read_json(path)
            reusable = (
                existing.get("status") == "ok"
                and existing.get("request_id") == request_id
                and existing.get("model_key") == args.model_key
                and existing.get("track_key") == TRACK_KEY
                and existing.get("protocol_sha256") == protocol_hash
                and existing.get("request_sha256") == canonical_sha256(request)
            )
            if reusable:
                completed[request_id] = existing
                continue
        pending.append(request)

    write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": (
                "gpt_oss_only_hard_v2_final_test_three_view_run_manifest_v5"
            ),
            "status": "running" if pending else "complete",
            "track_key": TRACK_KEY,
            "model_key": args.model_key,
            "mode": args.mode,
            "requests": len(requests),
            "pending_before_run": len(pending),
            "completed_before_run": len(completed),
            "protocol_sha256": protocol_hash,
            "max_new_tokens": max_new_tokens,
            "model": model,
            "runtime": {
                "vllm_version": observed_vllm,
                "visible_gpu_count": len(visible),
                "tensor_parallel_size": tensor_parallel_size,
                "VLLM_USE_FLASHINFER_SAMPLER": os.getenv(
                    "VLLM_USE_FLASHINFER_SAMPLER"
                ),
            },
            "scientific_resampling": False,
            "must_not_be_used_for_training": True,
        },
        overwrite=True,
    )

    if pending:
        from tqdm.auto import tqdm
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        auth: dict[str, Any] = {}
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token:
            auth["token"] = token
        tokenizer = AutoTokenizer.from_pretrained(
            model["model_id"],
            revision=model["tokenizer_revision"],
            trust_remote_code=bool(model.get("trust_remote_code", False)),
            **auth,
        )
        template_kwargs = dict(model.get("chat_template_kwargs", {}))
        rendered: dict[str, str] = {}
        input_lengths: dict[str, int] = {}
        max_input_tokens = int(protocol["evaluation"]["max_input_tokens"])
        for request in pending:
            request_id = str(request["request_id"])
            prompt = tokenizer.apply_chat_template(
                model_messages(request),
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            input_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if input_length > max_input_tokens:
                raise RuntimeError(
                    f"{request_id}: {input_length} input tokens exceed {max_input_tokens}"
                )
            rendered[request_id] = prompt
            input_lengths[request_id] = input_length

        llm = LLM(
            model=model["model_id"],
            tokenizer=model["model_id"],
            revision=model["model_revision"],
            tokenizer_revision=model["tokenizer_revision"],
            trust_remote_code=bool(model.get("trust_remote_code", False)),
            dtype=protocol["evaluation"]["dtype"],
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=float(model["gpu_memory_utilization"]),
            max_model_len=max_input_tokens + max_new_tokens,
            max_num_seqs=int(model["max_num_seqs"]),
            enable_prefix_caching=True,
            seed=int(protocol["evaluation"]["seed"]),
        )
        params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
            seed=int(protocol["evaluation"]["seed"]),
        )
        generate_signature = inspect.signature(llm.generate)
        request_batch_size = int(model["request_batch_size"])
        for start in tqdm(
            range(0, len(pending), request_batch_size),
            desc=f"GPT-OSS Hard V2 final-test/V5 {args.model_key} {args.mode}",
            unit="batch",
        ):
            batch = pending[start : start + request_batch_size]
            kwargs: dict[str, Any] = {"sampling_params": params}
            if accepts(generate_signature, "use_tqdm"):
                kwargs["use_tqdm"] = False
            outputs = list(
                llm.generate(
                    [rendered[str(request["request_id"])] for request in batch],
                    **kwargs,
                )
            )
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM output cardinality differs from request batch")
            for request, result in zip(batch, outputs):
                request_id = str(request["request_id"])
                choices = list(getattr(result, "outputs", []) or [])
                if len(choices) != 1:
                    raise RuntimeError(f"{request_id}: expected exactly one completion")
                choice = choices[0]
                text = str(getattr(choice, "text", "") or "").strip()
                token_ids = list(getattr(choice, "token_ids", []) or [])
                finish_reason = getattr(choice, "finish_reason", None)
                parsed = parse_compatible_response(text)
                answer = parsed["pred_answer"]
                view = str(request["view"])
                gold = str(request["gold_answer"])
                hard_target = str(request["hard_intended_target"])
                row = {
                    "schema_version": (
                        "gpt_oss_only_hard_v2_final_test_three_view_output_v5"
                    ),
                    "status": "ok" if text else "technical_error",
                    **request,
                    "track_key": TRACK_KEY,
                    "model_key": args.model_key,
                    "model_id": model["model_id"],
                    "model_revision_requested": model["model_revision"],
                    "tokenizer_revision_requested": model["tokenizer_revision"],
                    "protocol_sha256": protocol_hash,
                    "request_sha256": canonical_sha256(request),
                    "input_token_count": input_lengths[request_id],
                    "generated_token_count": len(token_ids),
                    "max_new_tokens": max_new_tokens,
                    "finish_reason": finish_reason,
                    "max_token_cutoff": finish_reason in {"length", "max_tokens"},
                    "generated_text": text,
                    **parsed,
                    "scientific_role": scientific_role_for_view(
                        answer,
                        view=view,
                        gold=gold,
                        hard_target=hard_target,
                    ),
                    "answer_correct": answer == gold,
                    "target_selected": (
                        answer == hard_target if view in {"clean", "hard"} else None
                    ),
                    "tensor_parallel_size": tensor_parallel_size,
                    "must_not_be_used_for_training": True,
                }
                write_json(shard_path(shards_dir, request_id), row, overwrite=True)
                completed[request_id] = row

    ordered: list[dict[str, Any]] = []
    for request in requests:
        request_id = str(request["request_id"])
        path = shard_path(shards_dir, request_id)
        if not path.is_file():
            raise RuntimeError(f"missing output shard: {request_id}")
        ordered.append(read_json(path))
    technical = [row["request_id"] for row in ordered if row.get("status") != "ok"]
    outputs_path = output_dir / "outputs.jsonl"
    write_jsonl(outputs_path, ordered, overwrite=True)
    final_manifest = {
        "schema_version": (
            "gpt_oss_only_hard_v2_final_test_three_view_run_manifest_v5"
        ),
        "status": "complete" if not technical else "technical_retry_required",
        "track_key": TRACK_KEY,
        "model_key": args.model_key,
        "mode": args.mode,
        "requests": len(requests),
        "max_new_tokens": max_new_tokens,
        "technical_failure_request_ids": technical,
        "protocol_sha256": protocol_hash,
        "bindings": {
            "protocol": {"path": str(args.protocol), "sha256": protocol_hash},
            "requests": {"path": str(args.requests), "sha256": sha256_file(args.requests)},
            "outputs": {"path": str(outputs_path), "sha256": sha256_file(outputs_path)},
        },
        "scientific_resampling": False,
        "must_not_be_used_for_training": True,
    }
    write_json(output_dir / "run_manifest.json", final_manifest, overwrite=True)
    print(f"Generation complete: {len(ordered)} responses")
    if technical:
        raise RuntimeError(
            f"technical retry required for {len(technical)} blank model outputs"
        )


if __name__ == "__main__":
    main()
