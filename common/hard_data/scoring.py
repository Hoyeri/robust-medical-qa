#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
from typing import Any
from collections.abc import Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.hard_data.prompting import (  # noqa: E402
    HARMONY_FINAL_ANSWER_PREFIX,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_messages,
)
from common.hard_data.runtime import (  # noqa: E402
    CHOICES,
    read_jsonl,
    sha256_file,
    sha256_text,
)


DEFAULT_MODEL = 'openai/gpt-oss-120b'
DEFAULT_REVISION = 'b5c939de8f754692c1647ca79fbf85e8c1e70f8a'


def as_token_ids(value: Any) -> list[int]:
    # Newer/custom Transformers chat templates may return a BatchEncoding
    # rather than the bare list returned by older releases.
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError(
                "tokenizer mapping output does not contain an input_ids field"
            )
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TypeError(f"unsupported tokenizer output type: {type(value).__name__}")
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("expected exactly one tokenized prompt")
        value = value[0]
    return [int(token_id) for token_id in value]


def longest_common_prefix(sequences: list[list[int]]) -> list[int]:
    if not sequences:
        return []
    limit = min(len(sequence) for sequence in sequences)
    index = 0
    while index < limit and len({sequence[index] for sequence in sequences}) == 1:
        index += 1
    return sequences[0][:index]


def build_final_answer_token_plan(tokenizer) -> dict[str, Any]:
    sequences = {
        choice: as_token_ids(
            tokenizer.encode(
                HARMONY_FINAL_ANSWER_PREFIX + choice,
                add_special_tokens=False,
            )
        )
        for choice in CHOICES
    }
    common_prefix = longest_common_prefix(list(sequences.values()))
    candidate_segments = {
        choice: sequence[len(common_prefix) :]
        for choice, sequence in sequences.items()
    }
    if not common_prefix:
        raise RuntimeError("Harmony final answer prefix tokenized to an empty prefix")
    if any(len(segment) != 1 for segment in candidate_segments.values()):
        raise RuntimeError(
            f"A-D are not single-token alternatives after the final prefix: {candidate_segments}"
        )
    candidate_token_ids = {
        choice: segment[0] for choice, segment in candidate_segments.items()
    }
    if len(set(candidate_token_ids.values())) != 4:
        raise RuntimeError(f"A-D token ids are not unique: {candidate_token_ids}")
    decoded_prefix = tokenizer.decode(common_prefix, skip_special_tokens=False)
    required_markers = (
        "<|start|>",
        "assistant",
        "<|channel|>",
        "final",
        "<|message|>",
    )
    if any(marker not in decoded_prefix for marker in required_markers):
        raise RuntimeError(
            "tokenized prefix does not preserve the forced Harmony final channel: "
            + repr(decoded_prefix)
        )
    if "<|channel|>analysis" in decoded_prefix:
        raise RuntimeError("forced answer prefix unexpectedly enters the analysis channel")
    return {
        "candidate_token_ids": candidate_token_ids,
        "common_prefix_ids": common_prefix,
        "decoded_common_prefix": decoded_prefix,
        "probe_token_ids": sequences,
    }


def render_user_turn_tokens(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": False,
        "reasoning_effort": "low",
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        effort = kwargs.pop("reasoning_effort")
        rendered = tokenizer.apply_chat_template(
            messages,
            chat_template_kwargs={"reasoning_effort": effort},
            **kwargs,
        )
    tokens = as_token_ids(rendered)
    decoded = tokenizer.decode(tokens, skip_special_tokens=False)
    if "<|channel|>analysis" in decoded:
        raise RuntimeError(
            "base user-turn rendering already contains an assistant analysis channel"
        )
    return tokens


def normalize(logprobs: dict[str, float]) -> dict[str, float]:
    maximum = max(logprobs.values())
    weights = {choice: math.exp(value - maximum) for choice, value in logprobs.items()}
    denominator = sum(weights.values())
    return {choice: weights[choice] / denominator for choice in CHOICES}


def unique_top1(logprobs: dict[str, float]) -> tuple[str | None, list[str]]:
    maximum = max(logprobs.values())
    winners = [choice for choice in CHOICES if logprobs[choice] == maximum]
    return (winners[0] if len(winners) == 1 else None), winners


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure GPT-OSS forced A-D preference with a Transformers direct "
            "forward pass, without vLLM generation or scheduling."
        )
    )
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_REVISION)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--attention-implementation",
        default="eager",
        choices=("sdpa", "eager"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if "gpt-oss-120b" not in args.model.lower():
        parser.error("this diagnostic is restricted to GPT-OSS-120B")
    if args.max_model_len <= 1:
        parser.error("max-model-len must be greater than one")
    return args


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def last_token_logits(model, input_ids):
    """Compute only the final-position LM-head logits when the API allows it."""
    try:
        output = model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        logits = output.logits
        if logits.ndim != 3 or logits.shape[1] != 1:
            raise RuntimeError(f"unexpected logits_to_keep output shape: {tuple(logits.shape)}")
        return logits[0, -1]
    except TypeError as exc:
        # Older compatible Transformers versions may not expose
        # logits_to_keep. Avoid materializing vocabulary logits for every
        # prompt position by running the backbone and applying lm_head only to
        # the final hidden state.
        if "logits_to_keep" not in str(exc):
            raise
        backbone = getattr(model, "model", None)
        lm_head = getattr(model, "lm_head", None)
        if backbone is None or lm_head is None:
            raise RuntimeError(
                "Transformers lacks logits_to_keep and the model does not expose model/lm_head"
            ) from exc
        hidden = backbone(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return lm_head(hidden[:, -1:, :])[0, -1]


def main() -> None:
    args = parse_args()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 before Python starts"
        )
    try:
        import accelerate
        import kernels
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import is_kernels_available, is_triton_available
    except ImportError as exc:
        raise RuntimeError(
            "Transformers direct MXFP4 scoring requires torch, transformers, "
            "accelerate, and kernels"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPT-OSS-120B direct scoring")
    if args.attention_implementation == "sdpa":
        raise RuntimeError(
            "this Transformers build does not support SDPA for GptOssForCausalLM; "
            "use --attention-implementation eager"
        )

    kernels_version = importlib.metadata.version("kernels")
    triton_version = importlib.metadata.version("triton")
    if not is_kernels_available():
        import transformers.utils.import_utils as import_utils

        minimum = getattr(import_utils, "KERNELS_MIN_VERSION", None)
        maximum = getattr(import_utils, "KERNELS_MAX_VERSION", None)
        raise RuntimeError(
            "Transformers does not accept the installed kernels package for MXFP4: "
            f"installed={kernels_version}, allowed_min={minimum}, "
            f"allowed_max={maximum}. Refusing the automatic BF16 dequantization "
            "because GPT-OSS-120B will not fit on one 96GB GPU."
        )
    if not is_triton_available("3.4.0"):
        raise RuntimeError(
            "MXFP4 requires Triton >=3.4.0: "
            f"installed={triton_version}"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    request_path = Path(args.requests)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    requests = read_jsonl(request_path)
    request_ids = [row["request_id"] for row in requests]
    if not requests:
        raise ValueError("request file is empty")
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("duplicate request ids")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
    )
    token_plan = build_final_answer_token_plan(tokenizer)
    rendered_requests: list[tuple[dict[str, Any], list[int], str]] = []
    for row in requests:
        prompt_tokens = (
            render_user_turn_tokens(tokenizer, build_messages(row))
            + token_plan["common_prefix_ids"]
        )
        if len(prompt_tokens) + 1 > args.max_model_len:
            raise RuntimeError(
                f"request exceeds max-model-len: {row['request_id']} {len(prompt_tokens)}"
            )
        rendered_requests.append(
            (
                row,
                prompt_tokens,
                sha256_text(",".join(map(str, prompt_tokens))),
            )
        )

    config = AutoConfig.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    quantization_config = getattr(config, "quantization_config", None) or {}
    if quantization_config.get("quant_method") != "mxfp4":
        raise RuntimeError(
            "expected the pinned GPT-OSS-120B checkpoint to declare MXFP4 quantization"
        )
    print(
        json.dumps(
            {
                "status": "cpu_prompt_preflight_complete_before_model_load",
                "requests": len(requests),
                "choice_token_ids": token_plan["candidate_token_ids"],
                "maximum_prompt_tokens": max(
                    len(prompt_tokens) for _, prompt_tokens, _ in rendered_requests
                ),
                "attention_implementation": args.attention_implementation,
                "deterministic_algorithms": True,
            },
            indent=2,
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
        dtype="auto",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        attn_implementation=args.attention_implementation,
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    candidate_ids = token_plan["candidate_token_ids"]

    if args.overwrite:
        records: dict[str, dict[str, Any]] = {}
    elif output_path.exists():
        existing_rows = read_jsonl(output_path)
        records = {row["request_id"]: row for row in existing_rows}
        if len(records) != len(existing_rows):
            raise ValueError("duplicate request ids in existing output")
    else:
        records = {}

    expected_ids = set(request_ids)
    if not set(records) <= expected_ids:
        raise ValueError('Foreign request IDs in saved scores')
    for row, _, prompt_hash in rendered_requests:
        saved = records.get(row['request_id'], {})
        if saved.get('status') != 'ok': continue
        if (saved.get('rendered_prompt_token_sha256') != prompt_hash
                or saved.get('model_revision') != args.model_revision
                or saved.get('model_id') != args.model
                or any(saved.get(k) != row.get(k) for k in ('source_id','candidate_id','gold','intended_target','role','condition'))):
            raise ValueError('Saved scoring input or model differs: ' + row['request_id'])
        values = saved.get('option_logprobs', {})
        if set(values) != set(CHOICES) or not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Invalid cached option scores')
        if saved.get('top1') != unique_top1(values)[0]:
            raise ValueError('Cached top1 does not match option scores')

    for index, (row, prompt_tokens, prompt_hash) in enumerate(rendered_requests, start=1):
        if records.get(row["request_id"], {}).get("status") == "ok":
            continue
        try:
            input_ids = torch.tensor(
                [prompt_tokens],
                dtype=torch.long,
                device=input_device,
            )
            with torch.inference_mode():
                logits = last_token_logits(model, input_ids)
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                option_logprobs = {
                    choice: float(logprobs[token_id].item())
                    for choice, token_id in candidate_ids.items()
                }
            if not all(math.isfinite(value) for value in option_logprobs.values()):
                raise ValueError("non-finite A-D logprob")
            top1, tied = unique_top1(option_logprobs)
            generated_choice = top1 if top1 is not None else sorted(tied)[0]
            record = {
                "analysis_generated": False,
                "candidate_id": row.get("candidate_id"),
                "choice_score_valid": True,
                "condition": row["condition"],
                "finish_reason": "direct_forward_forced_choice",
                "forced_channel": "final",
                "generated_choice": generated_choice,
                "generated_token_id": candidate_ids[generated_choice],
                "gold": row["gold"],
                "intended_target": row.get("intended_target"),
                "model_id": args.model,
                "model_revision": args.model_revision,
                "option_choice_scores": normalize(option_logprobs),
                "option_logprobs": option_logprobs,
                "prompt_token_count": len(prompt_tokens),
                "rendered_prompt_token_sha256": prompt_hash,
                "request_id": row["request_id"],
                "role": row["role"],
                "source_id": row["source_id"],
                "source_idx": row["source_idx"],
                "status": "ok",
                "top1": top1,
                "top1_tied_choices": tied if len(tied) > 1 else [],
            }
        except Exception as exc:
            record = {
                "request_id": row["request_id"],
                "source_idx": row["source_idx"],
                "role": row["role"],
                "candidate_id": row.get("candidate_id"),
                "status": "request_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        records[row["request_id"]] = record
        atomic_write_jsonl(
            output_path,
            [records[request_id] for request_id in request_ids if request_id in records],
        )
        if index == 1 or index % 10 == 0 or index == len(rendered_requests):
            print(f"Transformers direct scoring: {index}/{len(rendered_requests)}")

    final_rows = [records[request_id] for request_id in request_ids if request_id in records]
    ok = sum(row.get("status") == "ok" for row in final_rows)
    errors = len(requests) - ok
    manifest = {
        "schema_version": "gpt_oss_transformers_direct_score_manifest_v1",
        "status": "complete" if errors == 0 else "technical_failure",
        "counts": {"requests": len(requests), "ok": ok, "errors": errors},
        "measurement": {
            "description": "post-prompt pre-rationale forced A-D preference",
            "forced_channel": "harmony_final",
            "analysis_generated": False,
            "allowed_choices": list(CHOICES),
            "logprob_retrieval": "direct_last_token_log_softmax",
            "runtime_mode": "transformers_direct",
            "batch_size": 1,
            "temperature": 0.0,
            "max_tokens": 1,
            "seed": args.seed,
        },
        "determinism": {
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "two_independent_processes_required_for_repeatability_audit": True,
        },
        "model": {
            "backend": "transformers_direct_forward",
            "model_id": args.model,
            "model_revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision,
            "quantization": "mxfp4",
            "attention_implementation": args.attention_implementation,
            "torch_version": getattr(torch, "__version__", None),
            "transformers_version": getattr(transformers, "__version__", None),
            "accelerate_version": getattr(accelerate, "__version__", None),
            "kernels_version": kernels_version,
            "triton_version": triton_version,
            "model_dtype": str(getattr(model, "dtype", None)),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
        },
        "prompt": {
            "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
            "user_prompt_template_sha256": sha256_text(USER_PROMPT_TEMPLATE),
            "harmony_final_answer_prefix_sha256": sha256_text(
                HARMONY_FINAL_ANSWER_PREFIX
            ),
            "choice_token_ids": candidate_ids,
            "common_prefix_ids": token_plan["common_prefix_ids"],
        },
        "bindings": {
            "requests": {"path": str(request_path), "sha256": sha256_file(request_path)},
            "outputs": {"path": str(output_path), "sha256": sha256_file(output_path)},
            "scorer": {"path": str(Path(__file__)), "sha256": sha256_file(Path(__file__))},
            "forced_choice_helpers": {
                "path": str(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            },
        },
        "must_not_be_used_for_training": True,
    }
    atomic_write_json(manifest_path, manifest)
    print(f"Scoring complete: {ok}/{len(requests)}")
    if errors:
        raise RuntimeError(f"Transformers direct scoring failed for {errors} requests")


if __name__ == "__main__":
    main()
