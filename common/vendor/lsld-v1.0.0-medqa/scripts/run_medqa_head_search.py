from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from lsld_repro import paper_head
from lsld_repro.data import PromptExample


LETTERS = ("A", "B", "C", "D")


def build_prompt(question: str, options: dict[str, str]) -> str:
    lines = [question.strip(), ""]

    for letter in LETTERS:
        lines.append(f"{letter}. {options[letter]}")

    lines.extend(["", "Answer:"])
    return "\n".join(lines)


def make_prompt_example(row: dict, split: str, index: int) -> PromptExample:
    kwargs = {
        "example_id": str(
            row.get("question_id")
            or row.get("example_id")
            or f"{split}:{index:05d}"
        ),
        "relation": "medqa_hard",
        "setting": "hard",
        "no_context_prompt": build_prompt(
            row["clean_question"],
            row["options"],
        ),
        "with_context_prompt": build_prompt(
            row["distracted_question"],
            row["options"],
        ),
        "gold_text": row["gold_answer"],
        "distractor_text": row["intended_target"],
        "context_text": row.get("added_distractor", ""),
        "split": split,
    }

    signature = inspect.signature(PromptExample)

    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }

    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and name not in supported
    ]

    if missing:
        raise RuntimeError(
            f"PromptExample has unsupported required fields: {missing}"
        )

    return PromptExample(**supported)


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_medqa_splits(data_dir: Path) -> dict[str, list[PromptExample]]:
    result = {}

    for split in ("train", "dev", "test"):
        path = data_dir / f"{split}.jsonl"

        if not path.exists():
            raise FileNotFoundError(path)

        rows = load_jsonl(path)

        examples = [
            make_prompt_example(row, split, index)
            for index, row in enumerate(rows)
        ]

        result[split] = examples

    return result


def validate_splits(splits):
    counts = {name: len(rows) for name, rows in splits.items()}

    print("Split counts:", counts, flush=True)

    if counts != {"train": 89, "dev": 11, "test": 11}:
        raise RuntimeError(
            f"Expected 89/11/11 split, got {counts}"
        )

    all_ids = []

    for split in ("train", "dev", "test"):
        ids = [x.example_id for x in splits[split]]

        if len(ids) != len(set(ids)):
            raise RuntimeError(
                f"Duplicate example_id inside {split}"
            )

        all_ids.extend(ids)

    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError(
            "Same example appears in multiple splits"
        )


def validate_answer_tokens(tokenizer):
    print("=== answer token sanity check ===", flush=True)

    for letter in LETTERS:
        expected = tokenizer.encode(
            " " + letter,
            add_special_tokens=False,
        )

        released = paper_head.released_token_id(
            tokenizer,
            letter,
        )

        print(
            f"{letter}: tokenizer={expected}, "
            f"released_token_id={released}",
            flush=True,
        )

        if len(expected) != 1:
            raise RuntimeError(
                f"' {letter}' is not a single token: {expected}"
            )

        if expected[0] != released:
            raise RuntimeError(
                f"Token mismatch for {letter}: "
                f"{expected[0]} != {released}"
            )


def prepare_medqa_head_experiment(
    model,
    relation,
    *,
    seed: int,
    data_dir: Path,
):
    del relation, seed

    # Medical prompts are much longer than the original LSLD prompts.
    # During training, only logits at position -1 are used, so avoid
    # unembedding every sequence position.
    model._unembed_last_position_only = True

    splits = load_medqa_splits(data_dir)

    validate_splits(splits)
    validate_answer_tokens(model.tokenizer)

    datasets = SimpleNamespace(
        train=paper_head._to_hf_dataset(
            splits["train"],
            model.tokenizer,
        ),
        dev=paper_head._to_hf_dataset(
            splits["dev"],
            model.tokenizer,
        ),
        test=paper_head._to_hf_dataset(
            splits["test"],
            model.tokenizer,
        ),
    )

    model.setup_exp(
        datasets,
        paper_head.HEAD_BATCH_SIZE,
    )

    return splits


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
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

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {args.output_dir}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_prepare = paper_head.prepare_head_experiment
    original_epochs = paper_head.HEAD_EPOCHS
    original_batch_size = paper_head.HEAD_BATCH_SIZE

    def adapter(model, relation, *, seed):
        splits = prepare_medqa_head_experiment(
            model,
            relation,
            seed=seed,
            data_dir=args.data_dir,
        )
        model._gradient_accumulation_steps = args.grad_accum_steps
        return splits

    paper_head.prepare_head_experiment = adapter
    paper_head.HEAD_EPOCHS = args.epochs
    paper_head.HEAD_BATCH_SIZE = args.batch_size

    relation = SimpleNamespace(
        key="medqa_hard"
    )

    try:
        report = paper_head.train_paper_head_search(
            relation,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    finally:
        paper_head.prepare_head_experiment = original_prepare
        paper_head.HEAD_EPOCHS = original_epochs
        paper_head.HEAD_BATCH_SIZE = original_batch_size

    # Table 2는 country-capital reproduction용이므로
    # Medical 결과에서는 제거한다.
    report["test"].pop(
        "paper_table2_reference",
        None,
    )

    report["test"].pop(
        "difference_from_paper_table2",
        None,
    )

    report["medical_experiment"] = {
        "dataset": str(args.data_dir),
        "selection": (
            "clean top1 = gold AND "
            "distracted top1 = intended hard target"
        ),
        "train_n": 89,
        "dev_n": 11,
        "test_n": 11,
        "epochs": args.epochs,
        "seed": args.seed,
        "base_model": paper_head.HEAD_MODEL,
        "micro_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "paper_reference_batch_size": original_batch_size,
        "learning_rate": paper_head.HEAD_LEARNING_RATE,
        "sparsity_lambda": paper_head.HEAD_SPARSITY_LAMBDA,
        "temperature": paper_head.HEAD_TEMPERATURE,
    }

    paper_head.write_json(
        args.output_dir / "head_search_report.json",
        report,
    )

    print("\n===== MEDQA HEAD SEARCH COMPLETE =====")
    print(
        json.dumps(
            {
                "best_epoch": report["best_checkpoint"]["epoch_number"],
                "removed_heads": report["test"]["removed_heads"],
                "test": report["test"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
