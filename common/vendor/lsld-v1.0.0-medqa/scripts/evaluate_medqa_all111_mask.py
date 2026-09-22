import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lsld_repro import paper_head
import run_medqa_head_search as medqa


ANSWER_LABELS = ["A", "B", "C", "D"]


def load_best_mask(path: Path):
    data = json.loads(path.read_text())

    for key in ("keep_mask", "mask", "best_mask"):
        if key in data:
            mask = data[key]
            break
    else:
        raise RuntimeError(
            f"Mask not found in {path}. "
            f"Available keys: {list(data.keys())}"
        )

    mask = torch.tensor(mask, dtype=torch.float32)

    if mask.shape != (32, 32):
        raise RuntimeError(
            f"Expected mask shape (32, 32), got {tuple(mask.shape)}"
        )

    return data, mask


def get_value(batch, key, index):
    value = batch[key]

    if isinstance(value, (list, tuple)):
        return value[index]

    if torch.is_tensor(value):
        item = value[index]
        return item.item() if item.ndim == 0 else item

    return value


@torch.no_grad()
def score_condition(model, loader, prompt_field, answer_ids):
    rows = []

    for batch in loader:
        prompts = batch[prompt_field]

        logits = model(prompts)

        # Medical long-prompt optimization:
        # only final Answer: position is unembedded.
        last_logits = logits[:, -1, :]

        for i in range(last_logits.shape[0]):
            gold = str(get_value(batch, "gold_token", i))
            target = str(get_value(batch, "dstr_token", i))
            example_id = str(get_value(batch, "example_id", i))

            option_logits = {
                label: float(
                    last_logits[i, token_id].detach().cpu()
                )
                for label, token_id in answer_ids.items()
            }

            prediction = max(
                ANSWER_LABELS,
                key=lambda label: option_logits[label],
            )

            rows.append(
                {
                    "example_id": example_id,
                    "gold": gold,
                    "target": target,
                    "prediction": prediction,
                    "gold_logit": option_logits[gold],
                    "target_logit": option_logits[target],
                    "gold_minus_target_logit":
                        option_logits[gold] - option_logits[target],
                    "correct": prediction == gold,
                    "target_selected": prediction == target,
                    "other_wrong":
                        prediction != gold and prediction != target,
                    "option_logits": option_logits,
                }
            )

    return rows


def summarize(rows):
    n = len(rows)

    correct_count = sum(x["correct"] for x in rows)
    target_count = sum(x["target_selected"] for x in rows)
    other_count = sum(x["other_wrong"] for x in rows)

    return {
        "n": n,

        "mean_gold_logit":
            sum(x["gold_logit"] for x in rows) / n,

        "mean_target_logit":
            sum(x["target_logit"] for x in rows) / n,

        "mean_gold_minus_target_logit":
            sum(x["gold_minus_target_logit"] for x in rows) / n,

        "accuracy_count": correct_count,
        "accuracy": correct_count / n,

        "target_selected_count": target_count,
        "target_selected_rate": target_count / n,

        "other_wrong_count": other_count,
        "other_wrong_rate": other_count / n,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--best-mask",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _, mask = load_best_mask(args.best_mask)

    paper_head.HEAD_BATCH_SIZE = 1
    paper_head.set_public_seed(args.seed)

    model = paper_head.load_public_head_model(
        seed=args.seed
    )

    # Load original 89 / 11 / 11 splits
    splits = medqa.load_medqa_splits(
        args.data_dir
    )

    split_counts = {
        name: len(items)
        for name, items in splits.items()
    }

    all_examples = (
        list(splits["train"])
        + list(splits["dev"])
        + list(splits["test"])
    )

    ids = [x.example_id for x in all_examples]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate example IDs found after merging splits."
        )

    if len(all_examples) != 111:
        raise RuntimeError(
            f"Expected 111 examples, got {len(all_examples)}"
        )

    print(
        "Original split counts:",
        split_counts,
        flush=True,
    )

    print(
        f"Combined evaluation set: {len(all_examples)}",
        flush=True,
    )

    dataset = paper_head._to_hf_dataset(
        all_examples,
        model.tokenizer,
    )

    datasets = SimpleNamespace(
        train=dataset,
        dev=dataset,
        test=dataset,
    )

    # Batch 1 due to A5000 memory constraint
    model.setup_exp(
        datasets,
        1,
    )

    model._unembed_last_position_only = True

    answer_ids = {
        label: paper_head.released_token_id(
            model.tokenizer,
            label,
        )
        for label in ANSWER_LABELS
    }

    print(
        "Answer token IDs:",
        answer_ids,
        flush=True,
    )

    loader = model.test_dl

    # --------------------------------------------------
    # 1. Original model / Clean
    # 2. Original model / Hard
    # --------------------------------------------------
    model.set_mask_cfg(
        use_attention_head_mask=False,
        run_with_mask=False,
        use_deterministic_mask=True,
    )

    original_clean = score_condition(
        model,
        loader,
        "nctx_prompt",
        answer_ids,
    )

    original_hard = score_condition(
        model,
        loader,
        "wctx_prompt",
        answer_ids,
    )

    # --------------------------------------------------
    # 3. 219-head mask / Clean
    # 4. 219-head mask / Hard
    # --------------------------------------------------
    mask = mask.to(model.device)
    model.set_mask(mask)

    masked_clean = score_condition(
        model,
        loader,
        "nctx_prompt",
        answer_ids,
    )

    masked_hard = score_condition(
        model,
        loader,
        "wctx_prompt",
        answer_ids,
    )

    conditions = {
        "original_clean": original_clean,
        "original_hard": original_hard,
        "masked_clean": masked_clean,
        "masked_hard": masked_hard,
    }

    summaries = {
        name: summarize(rows)
        for name, rows in conditions.items()
    }

    report = {
        "split_counts": split_counts,
        "combined_n": len(all_examples),

        "total_heads": int(mask.numel()),
        "kept_heads": int(mask.sum().item()),
        "removed_heads":
            int(mask.numel() - mask.sum().item()),

        "logit_position": (
            "next-token logits at the final valid prompt position "
            "immediately after 'Answer:'"
        ),

        "accuracy_definition": (
            "restricted logit-based A/B/C/D argmax; "
            "no free-generation parsing"
        ),

        "answer_token_ids": answer_ids,

        "conditions": summaries,
    }

    output_json = (
        args.output_dir / "all111_mask_report.json"
    )

    output_json.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )

    detailed_path = (
        args.output_dir / "all111_mask_measurements.jsonl"
    )

    with detailed_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for i in range(len(all_examples)):
            row = {
                "example_id":
                    original_clean[i]["example_id"],
                "gold":
                    original_clean[i]["gold"],
                "target":
                    original_clean[i]["target"],
            }

            for condition, records in conditions.items():
                r = records[i]

                row[f"{condition}.prediction"] = (
                    r["prediction"]
                )
                row[f"{condition}.gold_logit"] = (
                    r["gold_logit"]
                )
                row[f"{condition}.target_logit"] = (
                    r["target_logit"]
                )
                row[
                    f"{condition}.gold_minus_target_logit"
                ] = r["gold_minus_target_logit"]

                row[f"{condition}.correct"] = (
                    r["correct"]
                )
                row[f"{condition}.target_selected"] = (
                    r["target_selected"]
                )
                row[f"{condition}.other_wrong"] = (
                    r["other_wrong"]
                )

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(
        "===== ALL 111 / 219-HEAD MASK EVALUATION ====="
    )
    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
