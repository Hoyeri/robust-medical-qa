import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lsld_repro import paper_head
import run_medqa_head_search as medqa
import scan_medqa_entrainment_heads as scan


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

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paper_head.HEAD_BATCH_SIZE = 1
    paper_head.set_public_seed(args.seed)

    model = paper_head.load_public_head_model(
        seed=args.seed
    )

    # 기존 89/11/11 파일은 그대로 두고 메모리에서만 합친다.
    original_splits = medqa.load_medqa_splits(
        args.data_dir
    )

    all_examples = (
        list(original_splits["train"])
        + list(original_splits["dev"])
        + list(original_splits["test"])
    )

    ids = [x.example_id for x in all_examples]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate example IDs found after merging splits."
        )

    print(
        f"Combined examples: {len(all_examples)}",
        flush=True,
    )

    if len(all_examples) != 111:
        raise RuntimeError(
            f"Expected 111 examples, got {len(all_examples)}"
        )

    # 111개 전체를 하나의 DataLoader로 사용한다.
    all_dataset = paper_head._to_hf_dataset(
        all_examples,
        model.tokenizer,
    )

    datasets = SimpleNamespace(
        train=all_dataset,
        dev=all_dataset,
        test=all_dataset,
    )

    model.setup_exp(
        datasets,
        1,
    )

    # 긴 Medical prompt용 메모리 최적화
    model._unembed_last_position_only = True

    answer_ids = {
        label: paper_head.released_token_id(
            model.tokenizer,
            label,
        )
        for label in scan.ANSWER_LABELS
    }

    print(
        "Answer token IDs:",
        answer_ids,
        flush=True,
    )

    # Llama-3.1-8B의 32 x 32 = 1024 heads 전체
    candidates = [
        (layer, head)
        for layer in range(int(model.cfg.n_layers))
        for head in range(int(model.cfg.n_heads))
    ]

    print(
        f"Total heads to scan: {len(candidates)}",
        flush=True,
    )

    if len(candidates) != 1024:
        raise RuntimeError(
            f"Expected 1024 heads, got {len(candidates)}"
        )

    result = scan.evaluate_split(
        model,
        "train",
        candidates,
        answer_ids,
        args.output_dir,
    )

    # 기존 scanner가 train 이름으로 저장하므로
    # all111 이름으로 바꾼다.
    rename_pairs = [
        (
            "train_head_scores.json",
            "all111_head_scores.json",
        ),
        (
            "train_head_scores.csv",
            "all111_head_scores.csv",
        ),
        (
            "train_per_example.jsonl",
            "all111_per_example.jsonl",
        ),
    ]

    for src_name, dst_name in rename_pairs:
        src = args.output_dir / src_name
        dst = args.output_dir / dst_name

        if src.exists():
            src.replace(dst)

    summary = {
        "n_examples": len(all_examples),
        "n_heads": len(candidates),
        "baseline_entrainment_effect":
            result["baseline_entrainment_effect"],
        "scoring_definition": (
            "E_all - E_single_head_ablation, "
            "E=(hard-gold)_distracted"
            "-(hard-gold)_clean"
        ),
        "top20": result["scores"][:20],
    }

    (
        args.output_dir / "all111_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "===== ALL 111 x 1024 HEAD SCAN COMPLETE =====",
        flush=True,
    )


if __name__ == "__main__":
    main()
