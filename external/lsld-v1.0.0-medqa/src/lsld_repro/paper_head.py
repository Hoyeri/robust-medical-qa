"""Paper-first entrainment-head training and evaluation.

The differentiable mask implementation is vendored byte-for-byte from the
authors' public repository.  This module only supplies the missing experiment
driver: result-producing data reconstruction, checkpoint selection, and a
test-only final report with correct whole-dataset aggregation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .data import PromptExample, Relation
from .io_utils import environment_metadata, write_json, write_jsonl
from .paper_head_data import (
    build_paper_head_search_splits,
    expected_country_capital_counts,
)
from .protocol import (
    A5000_DTYPE,
    HEAD_BATCH_SIZE,
    HEAD_EPOCHS,
    HEAD_LEARNING_RATE,
    HEAD_MODEL,
    HEAD_SELECTION_WEIGHT,
    HEAD_SPARSITY_LAMBDA,
    HEAD_TEMPERATURE,
    protocol_manifest,
)
from .tokenization import released_first_wordpiece

if TYPE_CHECKING:
    import torch


TABLE2_REFERENCE = {
    "removed_heads": 36,
    "all_heads_no_context": {
        "gold_logit": 19.51,
        "distractor_logit": 8.75,
        "gold_minus_distractor_logit": 10.76,
        "gold_rank": 1.00,
        "distractor_rank": 1756.7,
    },
    "all_heads_with_context": {
        "gold_logit": 20.68,
        "distractor_logit": 12.99,
        "gold_minus_distractor_logit": 7.69,
        "gold_rank": 1.00,
        "distractor_rank": 37.5,
    },
    "masked_no_context": {
        "gold_logit": 19.49,
        "distractor_logit": 7.87,
        "gold_minus_distractor_logit": 11.62,
        "gold_rank": 1.00,
        "distractor_rank": 1707.3,
    },
    "masked_with_context": {
        "gold_logit": 21.21,
        "distractor_logit": 8.01,
        "gold_minus_distractor_logit": 13.20,
        "gold_rank": 1.00,
        "distractor_rank": 1289.6,
    },
}


def set_public_seed(seed: int) -> None:
    """Apply the three seeds set by the public head-analysis module."""

    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def install_a5000_mask_dtype_adapter() -> None:
    """Keep FP32 mask logits while multiplying FP16 head outputs safely.

    The vendored public mask uses ``einsum(z, mask)`` and assumes both are
    float32.  Our sole A5000 model-dtype deviation makes ``z`` float16, while
    keeping mask logits float32 is important for AdamW at learning rate 1.0.
    This adapter performs the identical head-wise multiplication after casting
    only the sampled mask for that operation.  The cast remains differentiable
    back to the FP32 mask parameter.
    """

    import torch

    from .official_code.head_search.circuit_lms import transformer_blocks

    def a5000_apply_mask(x: Any, mask_logits: Any, deterministic: bool = False):
        if deterministic:
            sampled = torch.where(mask_logits > 0.0, 1.0, 0.0)
        else:
            sampled = transformer_blocks.gumbel_sigmoid(mask_logits)
        sampled = sampled.to(device=x.device, dtype=x.dtype)
        return x * sampled.view(1, 1, -1, 1)

    transformer_blocks.apply_mask = a5000_apply_mask


def released_token_id(tokenizer: Any, token: str) -> int:
    """Use the public Llama-3.1 first-wordpiece rule verbatim."""

    return released_first_wordpiece(tokenizer, token).token_id


def _to_hf_dataset(examples: Sequence[PromptExample], tokenizer: Any):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "example_id": [item.example_id for item in examples],
            "wctx_prompt": [item.with_context_prompt for item in examples],
            "nctx_prompt": [item.no_context_prompt for item in examples],
            "dstr_token": [item.distractor_text for item in examples],
            "dstr_token_idx": [
                released_token_id(tokenizer, item.distractor_text)
                for item in examples
            ],
            "gold_token": [item.gold_text for item in examples],
            "gold_token_idx": [
                released_token_id(tokenizer, item.gold_text) for item in examples
            ],
        }
    ).with_format("torch")


def load_public_head_model(*, seed: int):
    """Load the public TransformerLens model with the sole A5000 deviation."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Head search requires a CUDA GPU")
    from .official_code.head_search.head_analysis import HeadAnalysis

    # Importing the released module itself resets torch/random/numpy to zero.
    # Re-apply the requested run seed after that import so Appendix-D repeats
    # are genuinely independent while seed=0 remains byte-for-byte equivalent
    # to the released initialization.
    set_public_seed(seed)
    install_a5000_mask_dtype_adapter()

    model = HeadAnalysis.from_pretrained(
        HEAD_MODEL,
        dtype=A5000_DTYPE,
        device="cuda",
        n_devices=1,
        default_padding_side="right",
    )
    mask_cfg = SimpleNamespace(
        use_attention_head_mask=True,
        use_deterministic_mask=False,
        run_with_mask=False,
        mask=None,
    )
    model.setup_mask_param(mask_cfg)
    if model.cfg.n_layers != 32 or model.cfg.n_heads != 32:
        raise RuntimeError(
            f"Expected 32x32 heads, got {model.cfg.n_layers}x{model.cfg.n_heads}"
        )
    if model.cfg.dtype != torch.float16:
        raise RuntimeError(f"Expected A5000 model dtype float16, got {model.cfg.dtype}")
    mask_dtypes = {parameter.dtype for parameter in model.mask_dict.values()}
    if mask_dtypes != {torch.float32}:
        raise RuntimeError(f"Expected float32 mask logits, got {mask_dtypes}")
    return model


def prepare_head_experiment(model: Any, relation: Relation, *, seed: int):
    """Build the audited splits and attach exact public DataLoaders."""

    splits = build_paper_head_search_splits(relation, seed=seed)
    datasets = SimpleNamespace(
        train=_to_hf_dataset(splits["train"], model.tokenizer),
        dev=_to_hf_dataset(splits["dev"], model.tokenizer),
        test=_to_hf_dataset(splits["test"], model.tokenizer),
    )
    model.setup_exp(datasets, HEAD_BATCH_SIZE)
    return splits


def _mask_matrix(model: Any):
    import torch

    return torch.stack(
        [torch.where(mask > 0.0, 1.0, 0.0) for mask in model.mask_dict.values()]
    )


def _summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Cannot summarize an empty set of rows")
    fields = (
        "gold_logit",
        "distractor_logit",
        "gold_minus_distractor_logit",
        "gold_probability",
        "distractor_probability",
        "gold_rank",
        "distractor_rank",
    )
    result: dict[str, float | int] = {"n": len(rows)}
    for field in fields:
        result[field] = float(np.mean([float(row[field]) for row in rows]))
    return result


def released_last_batch(
    rows: Sequence[dict[str, Any]], batch_size: int = HEAD_BATCH_SIZE
) -> Sequence[dict[str, Any]]:
    """Reproduce the public evaluate() aggregation bug used for selection."""

    if not rows:
        raise ValueError("Development rows must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    remainder = len(rows) % batch_size
    last_size = remainder if remainder else min(batch_size, len(rows))
    return rows[-last_size:]


def paper_epoch_selection_score(dev_gap: float, kept_heads: int) -> float:
    """Paper footnote 5: logit gap plus number of kept heads times 0.1."""

    return float(dev_gap) + int(kept_heads) * HEAD_SELECTION_WEIGHT


def removed_head_coordinates(mask: Any) -> list[dict[str, int]]:
    """Return zero-indexed coordinates for every deterministic zero mask."""

    result: list[dict[str, int]] = []
    for layer, row in enumerate(mask):
        for head, value in enumerate(row):
            scalar = float(value.item()) if hasattr(value, "item") else float(value)
            if scalar == 0.0:
                result.append({"layer": layer, "head": head})
    return result


def _score_loader(
    model: Any,
    dataloader: Any,
    *,
    prompt_field: str,
) -> list[dict[str, Any]]:
    """Score every row at its last valid token, as in public evaluation."""

    import torch

    output: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            prompts = batch[prompt_field]
            encoded = model.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = encoded["input_ids"].to(model.device)
            logits = model(input_ids)
            lengths = encoded["attention_mask"].sum(dim=1).to(logits.device)
            rows = torch.arange(len(prompts), device=logits.device)
            last = logits[rows, lengths - 1, :]
            probabilities = last.softmax(dim=-1)
            gold_ids = batch["gold_token_idx"].to(logits.device)
            distractor_ids = batch["dstr_token_idx"].to(logits.device)
            gold_logits = last[rows, gold_ids]
            distractor_logits = last[rows, distractor_ids]
            gold_probs = probabilities[rows, gold_ids]
            distractor_probs = probabilities[rows, distractor_ids]
            gold_ranks = (last > gold_logits[:, None]).sum(dim=1) + 1
            distractor_ranks = (last > distractor_logits[:, None]).sum(dim=1) + 1
            for index in range(len(prompts)):
                output.append(
                    {
                        "example_id": batch["example_id"][index],
                        "gold_token": batch["gold_token"][index],
                        "distractor_token": batch["dstr_token"][index],
                        "gold_logit": float(gold_logits[index].float().cpu()),
                        "distractor_logit": float(
                            distractor_logits[index].float().cpu()
                        ),
                        "gold_minus_distractor_logit": float(
                            (gold_logits[index] - distractor_logits[index])
                            .float()
                            .cpu()
                        ),
                        "gold_probability": float(gold_probs[index].float().cpu()),
                        "distractor_probability": float(
                            distractor_probs[index].float().cpu()
                        ),
                        "gold_rank": int(gold_ranks[index].cpu()),
                        "distractor_rank": int(distractor_ranks[index].cpu()),
                    }
                )
    return output


def _current_dev_metrics(model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return both public selection aggregation and correct full-dev metrics."""

    model.set_mask_cfg(
        use_attention_head_mask=True,
        run_with_mask=False,
        use_deterministic_mask=True,
    )
    rows = _score_loader(model, model.dev_dl, prompt_field="wctx_prompt")
    return _summarize_rows(released_last_batch(rows)), _summarize_rows(rows)


def _paired_result(after: Sequence[float], before: Sequence[float]) -> dict[str, float]:
    from scipy.stats import ttest_rel

    differences = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    test = ttest_rel(after, before)
    return {
        "mean_difference": float(differences.mean()),
        "paired_t": float(test.statistic),
        "paired_p": float(test.pvalue),
    }


def evaluate_mask(model: Any, mask: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate four test conditions without the released aggregation bug."""

    import torch

    model.set_mask_cfg(
        use_attention_head_mask=False,
        run_with_mask=False,
        use_deterministic_mask=True,
    )
    all_no = _score_loader(model, model.test_dl, prompt_field="nctx_prompt")
    all_with = _score_loader(model, model.test_dl, prompt_field="wctx_prompt")

    mask = torch.as_tensor(mask, dtype=torch.float32, device=model.device)
    model.set_mask(mask)
    masked_no = _score_loader(model, model.test_dl, prompt_field="nctx_prompt")
    masked_with = _score_loader(model, model.test_dl, prompt_field="wctx_prompt")

    detailed: list[dict[str, Any]] = []
    for index in range(len(all_no)):
        row: dict[str, Any] = {
            "example_id": all_no[index]["example_id"],
            "gold_token": all_no[index]["gold_token"],
            "distractor_token": all_no[index]["distractor_token"],
        }
        for name, records in (
            ("all_heads_no_context", all_no),
            ("all_heads_with_context", all_with),
            ("masked_no_context", masked_no),
            ("masked_with_context", masked_with),
        ):
            for key, value in records[index].items():
                if key not in {"example_id", "gold_token", "distractor_token"}:
                    row[f"{name}.{key}"] = value
        row["all_heads.context_distractor_effect"] = float(
            row["all_heads_with_context.distractor_logit"]
        ) - float(row["all_heads_no_context.distractor_logit"])
        row["masked.context_distractor_effect"] = float(
            row["masked_with_context.distractor_logit"]
        ) - float(row["masked_no_context.distractor_logit"])
        detailed.append(row)

    keep_count = int(mask.sum().item())
    report = {
        "total_heads": int(mask.numel()),
        "kept_heads": keep_count,
        "removed_heads": int(mask.numel()) - keep_count,
        "all_heads_no_context": _summarize_rows(all_no),
        "all_heads_with_context": _summarize_rows(all_with),
        "masked_no_context": _summarize_rows(masked_no),
        "masked_with_context": _summarize_rows(masked_with),
        "paired_tests": {
            "distractor_rank_masked_minus_all_heads_with_context": _paired_result(
                [row["distractor_rank"] for row in masked_with],
                [row["distractor_rank"] for row in all_with],
            ),
            "gold_minus_distractor_gap_masked_minus_all_heads_with_context": (
                _paired_result(
                    [row["gold_minus_distractor_logit"] for row in masked_with],
                    [row["gold_minus_distractor_logit"] for row in all_with],
                )
            ),
            "context_distractor_effect_masked_minus_all_heads": _paired_result(
                [row["masked.context_distractor_effect"] for row in detailed],
                [row["all_heads.context_distractor_effect"] for row in detailed],
            ),
        },
    }
    return report, detailed


def _comparison_to_table2(report: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "removed_heads_difference": int(report["removed_heads"])
        - int(TABLE2_REFERENCE["removed_heads"])
    }
    for condition in (
        "all_heads_no_context",
        "all_heads_with_context",
        "masked_no_context",
        "masked_with_context",
    ):
        observed = report[condition]
        reference = TABLE2_REFERENCE[condition]
        comparison[condition] = {
            f"{metric}_difference": float(observed[metric]) - float(reference[metric])
            for metric in reference
        }
    return comparison


def train_paper_head_search(
    relation: Relation,
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the fixed 500-epoch protocol and evaluate its selected mask once."""

    import torch
    import torch.nn.functional as F

    set_public_seed(seed)
    model = load_public_head_model(seed=seed)
    splits = prepare_head_experiment(model, relation, seed=seed)
    counts = {name: len(items) for name, items in splits.items()}
    if relation.key == "country_capital_city" and seed == 0:
        expected = expected_country_capital_counts()
        if counts != expected:
            raise RuntimeError(f"Expected {expected}, generated {counts}")

    write_jsonl(output_dir / "train_examples.jsonl", splits["train"])
    write_jsonl(output_dir / "dev_examples.jsonl", splits["dev"])
    write_jsonl(output_dir / "test_examples.jsonl", splits["test"])
    write_json(
        output_dir / "run_config.json",
        {
            "seed": seed,
            "relation": relation.key,
            "counts": counts,
            "model_dtype": str(model.cfg.dtype),
            "mask_logit_dtype": str(next(iter(model.mask_dict.values())).dtype),
            "tokenizer_revision": model.tokenizer.init_kwargs.get("_commit_hash"),
            "protocol": protocol_manifest(),
            "environment": environment_metadata(),
        },
    )

    optimizer = torch.optim.AdamW(model.mask_dict.values(), lr=HEAD_LEARNING_RATE)
    best_score = -float("inf")
    best: dict[str, Any] | None = None
    history_path = output_dir / "history.jsonl"

    with history_path.open("w", encoding="utf-8") as history:
        for epoch_index in range(HEAD_EPOCHS):
            model.train()
            model.set_mask_cfg(
                use_attention_head_mask=True,
                run_with_mask=False,
                use_deterministic_mask=False,
            )
            epoch_losses: list[float] = []
            epoch_effect_losses: list[float] = []
            epoch_keep_probabilities: list[float] = []

            # The released DataLoader is deterministic and does not shuffle.
            # Default grad_accum_steps=1 preserves the released reproduction.
            # Medical long-prompt runs may opt into microbatch accumulation.
            grad_accum_steps = int(
                getattr(model, "_gradient_accumulation_steps", 1)
            )
            if grad_accum_steps < 1:
                raise ValueError(
                    f"gradient accumulation steps must be >= 1, got {grad_accum_steps}"
                )

            total_microbatches = len(model.train_dl)
            optimizer_steps = 0
            optimizer.zero_grad(set_to_none=True)

            if epoch_index == 0 and grad_accum_steps > 1:
                expected_steps = (
                    total_microbatches + grad_accum_steps - 1
                ) // grad_accum_steps
                print(
                    f"gradient_accumulation_steps={grad_accum_steps} "
                    f"microbatches_per_epoch={total_microbatches} "
                    f"optimizer_steps_per_epoch={expected_steps}",
                    flush=True,
                )

            for micro_index, batch in enumerate(model.train_dl):
                prompts = batch["wctx_prompt"]
                batch_rows = torch.arange(len(prompts))
                logits = model(prompts)
                # Intentional public behavior: -1 on a right-padded string batch.
                last_token_logits = logits[batch_rows, -1, :]
                gold_ids = batch["gold_token_idx"]
                distractor_ids = batch["dstr_token_idx"]
                gold_logits = last_token_logits[batch_rows, gold_ids]
                distractor_logits = last_token_logits[batch_rows, distractor_ids]
                binary_logits = torch.stack([gold_logits, distractor_logits], dim=-1)
                effect_loss = F.cross_entropy(
                    binary_logits,
                    torch.zeros(len(prompts), dtype=torch.long, device=binary_logits.device),
                )
                keep_probability = model.sparsity_loss()
                # Operational public loss. See protocol audit for paper-sign conflict.
                loss = effect_loss - HEAD_SPARSITY_LAMBDA * keep_probability

                window_start = (
                    micro_index // grad_accum_steps
                ) * grad_accum_steps
                window_size = min(
                    grad_accum_steps,
                    total_microbatches - window_start,
                )

                (loss / window_size).backward()

                should_step = (
                    (micro_index + 1) % grad_accum_steps == 0
                    or (micro_index + 1) == total_microbatches
                )

                if should_step:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1

                epoch_losses.append(float(loss.detach().cpu()))
                epoch_effect_losses.append(float(effect_loss.detach().cpu()))
                epoch_keep_probabilities.append(
                    float(keep_probability.detach().cpu())
                )

            selection_dev, full_dev = _current_dev_metrics(model)
            mask = _mask_matrix(model).detach()
            kept_heads = int(mask.sum().item())
            removed_heads = int(mask.numel()) - kept_heads
            paper_score = paper_epoch_selection_score(
                float(selection_dev["gold_minus_distractor_logit"]), kept_heads
            )
            record = {
                "epoch_index": epoch_index,
                "epoch_number": epoch_index + 1,
                "loss": float(np.mean(epoch_losses)),
                "effect_loss": float(np.mean(epoch_effect_losses)),
                "mean_keep_probability": float(np.mean(epoch_keep_probabilities)),
                "kept_heads": kept_heads,
                "removed_heads": removed_heads,
                "selection_dev": selection_dev,
                "full_dev_diagnostic": full_dev,
                "paper_selection_score": paper_score,
                "keep_mask": mask.cpu().tolist(),
                "removed_head_coordinates": removed_head_coordinates(mask),
            }
            history.write(json.dumps(record, ensure_ascii=False) + "\n")
            history.flush()
            print(
                f"epoch_index={epoch_index} epoch_number={epoch_index + 1} "
                f"loss={record['loss']:.4f} "
                f"selection_gap={float(selection_dev['gold_minus_distractor_logit']):.4f} "
                f"full_dev_gap={float(full_dev['gold_minus_distractor_logit']):.4f} "
                f"kept={kept_heads} removed={removed_heads} "
                f"paper_score={paper_score:.4f}",
                flush=True,
            )

            if paper_score > best_score:
                best_score = paper_score
                best = record
                write_json(output_dir / "best_mask.json", record)

    if best is None:
        raise RuntimeError("No checkpoint was selected")
    report, detailed = evaluate_mask(model, best["keep_mask"])
    report["paper_table2_reference"] = TABLE2_REFERENCE
    report["difference_from_paper_table2"] = _comparison_to_table2(report)
    final = {"best_checkpoint": best, "test": report}
    write_jsonl(output_dir / "test_measurements.jsonl", detailed)
    write_json(output_dir / "head_search_report.json", final)
    return final


def load_official_reference_mask(path: Path) -> tuple[list[list[float]], dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    mask = payload["dev"]["mask"]
    return mask, payload


def evaluate_official_reference_mask(
    relation: Relation,
    *,
    seed: int,
    mask_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate the authors' published 36-head mask with full test aggregation."""

    set_public_seed(seed)
    model = load_public_head_model(seed=seed)
    splits = prepare_head_experiment(model, relation, seed=seed)
    mask, source_payload = load_official_reference_mask(mask_path)
    report, detailed = evaluate_mask(model, mask)
    report["paper_table2_reference"] = TABLE2_REFERENCE
    report["difference_from_paper_table2"] = _comparison_to_table2(report)
    payload = {
        "source_mask": str(mask_path),
        "source_dev_n_inferred_from_correct_and_acc": int(
            round(
                float(source_payload["dev"]["correct"])
                / float(source_payload["dev"]["acc"])
            )
        ),
        "split_counts": {name: len(items) for name, items in splits.items()},
        "model_dtype": str(model.cfg.dtype),
        "mask_logit_dtype": str(next(iter(model.mask_dict.values())).dtype),
        "tokenizer_revision": model.tokenizer.init_kwargs.get("_commit_hash"),
        "test": report,
        "protocol": protocol_manifest(),
        "environment": environment_metadata(),
    }
    write_jsonl(output_dir / "test_measurements.jsonl", detailed)
    write_json(output_dir / "reference_mask_report.json", payload)
    return payload
