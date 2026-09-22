import argparse
import csv
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


def load_candidates(path: Path):
    data = json.loads(path.read_text())

    if "removed_head_coordinates" not in data:
        raise RuntimeError(
            f"removed_head_coordinates not found in {path}"
        )

    coords = [
        (int(x["layer"]), int(x["head"]))
        for x in data["removed_head_coordinates"]
    ]

    return data, coords


def extract_value(batch, key, index):
    value = batch[key]

    if isinstance(value, (list, tuple)):
        return value[index]

    if torch.is_tensor(value):
        item = value[index]
        return item.item() if item.ndim == 0 else item

    return value


@torch.no_grad()
def score_loader(model, loader, prompt_field, answer_ids):
    records = []

    for batch in loader:
        prompts = batch[prompt_field]

        logits = model(prompts)

        # Medical long-prompt optimization:
        # only final answer position is unembedded.
        last_logits = logits[:, -1, :]

        batch_size = last_logits.shape[0]

        for i in range(batch_size):
            scores = {
                label: float(
                    last_logits[i, token_id].detach().cpu()
                )
                for label, token_id in answer_ids.items()
            }

            gold = str(extract_value(batch, "gold_token", i))
            hard = str(extract_value(batch, "dstr_token", i))
            example_id = str(
                extract_value(batch, "example_id", i)
            )

            prediction = max(
                ANSWER_LABELS,
                key=lambda label: scores[label],
            )

            gold_hard_gap = scores[gold] - scores[hard]

            records.append(
                {
                    "example_id": example_id,
                    "gold": gold,
                    "hard": hard,
                    "prediction": prediction,
                    "gold_hard_gap": gold_hard_gap,
                }
            )

    return records


def paired_metrics(clean_rows, distracted_rows):
    if len(clean_rows) != len(distracted_rows):
        raise RuntimeError("clean/distracted size mismatch")

    rows = []

    for clean, distracted in zip(clean_rows, distracted_rows):
        if clean["example_id"] != distracted["example_id"]:
            raise RuntimeError("example order mismatch")

        # H-G shift caused by distractor.
        entrainment_effect = (
            -distracted["gold_hard_gap"]
            + clean["gold_hard_gap"]
        )

        rows.append(
            {
                "example_id": clean["example_id"],
                "gold": clean["gold"],
                "hard": clean["hard"],
                "clean_prediction": clean["prediction"],
                "distracted_prediction": distracted["prediction"],
                "clean_gap": clean["gold_hard_gap"],
                "distracted_gap": distracted["gold_hard_gap"],
                "entrainment_effect": entrainment_effect,
            }
        )

    return rows


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize_single_head(
    baseline_rows,
    ablated_rows,
    layer,
    head,
):
    if len(baseline_rows) != len(ablated_rows):
        raise RuntimeError("baseline/ablation size mismatch")

    per_example = []

    for base, abl in zip(baseline_rows, ablated_rows):
        if base["example_id"] != abl["example_id"]:
            raise RuntimeError("baseline/ablation order mismatch")

        clean_gap_change = (
            abl["clean_gap"] - base["clean_gap"]
        )

        distracted_gap_change = (
            abl["distracted_gap"]
            - base["distracted_gap"]
        )

        # Equivalent to:
        # E_all - E_ablation
        entrainment_score = (
            base["entrainment_effect"]
            - abl["entrainment_effect"]
        )

        per_example.append(
            {
                "example_id": base["example_id"],
                "entrainment_score": entrainment_score,
                "clean_gap_change": clean_gap_change,
                "distracted_gap_change": distracted_gap_change,
                "clean_correct_after_ablation":
                    abl["clean_prediction"] == abl["gold"],
                "distracted_correct_after_ablation":
                    abl["distracted_prediction"] == abl["gold"],
                "hard_selected_after_ablation":
                    abl["distracted_prediction"] == abl["hard"],
            }
        )

    n = len(per_example)

    result = {
        "layer": layer,
        "head": head,
        "head_name": f"L{layer}H{head}",
        "n": n,

        "entrainment_score": mean(
            [x["entrainment_score"] for x in per_example]
        ),

        "clean_gap_change": mean(
            [x["clean_gap_change"] for x in per_example]
        ),

        "distracted_gap_change": mean(
            [x["distracted_gap_change"] for x in per_example]
        ),

        "clean_accuracy_after_ablation": mean(
            [
                float(x["clean_correct_after_ablation"])
                for x in per_example
            ]
        ),

        "distracted_accuracy_after_ablation": mean(
            [
                float(x["distracted_correct_after_ablation"])
                for x in per_example
            ]
        ),

        "hard_target_rate_after_ablation": mean(
            [
                float(x["hard_selected_after_ablation"])
                for x in per_example
            ]
        ),
    }

    return result, per_example


def loader_for_split(model, split):
    if split == "train":
        return model.train_dl
    if split == "dev":
        return model.dev_dl
    if split == "test":
        return model.test_dl

    raise ValueError(split)


def evaluate_split(
    model,
    split,
    candidates,
    answer_ids,
    output_dir,
):
    loader = loader_for_split(model, split)

    print(f"\n===== {split.upper()} BASELINE =====", flush=True)

    # Intact model
    model.set_mask_cfg(
        use_attention_head_mask=False,
        run_with_mask=False,
        use_deterministic_mask=True,
    )

    baseline_clean = score_loader(
        model,
        loader,
        "nctx_prompt",
        answer_ids,
    )

    baseline_distracted = score_loader(
        model,
        loader,
        "wctx_prompt",
        answer_ids,
    )

    baseline_rows = paired_metrics(
        baseline_clean,
        baseline_distracted,
    )

    baseline_effect = mean(
        [x["entrainment_effect"] for x in baseline_rows]
    )

    print(
        f"{split}: n={len(baseline_rows)} "
        f"baseline_entrainment_effect={baseline_effect:.6f}",
        flush=True,
    )

    results = []
    detailed_path = output_dir / f"{split}_per_example.jsonl"

    with detailed_path.open("w", encoding="utf-8") as detailed_file:
        for idx, (layer, head) in enumerate(candidates, start=1):

            # Single-head ablation from intact model.
            # Every other head remains enabled.
            mask = torch.ones(
                (
                    int(model.cfg.n_layers),
                    int(model.cfg.n_heads),
                ),
                dtype=torch.float32,
                device=model.device,
            )

            mask[layer, head] = 0.0
            model.set_mask(mask)

            ablated_clean = score_loader(
                model,
                loader,
                "nctx_prompt",
                answer_ids,
            )

            ablated_distracted = score_loader(
                model,
                loader,
                "wctx_prompt",
                answer_ids,
            )

            ablated_rows = paired_metrics(
                ablated_clean,
                ablated_distracted,
            )

            summary, per_example = summarize_single_head(
                baseline_rows,
                ablated_rows,
                layer,
                head,
            )

            results.append(summary)

            for row in per_example:
                detailed_file.write(
                    json.dumps(
                        {
                            "split": split,
                            "layer": layer,
                            "head": head,
                            "head_name": f"L{layer}H{head}",
                            **row,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            print(
                f"[{idx:03d}/{len(candidates):03d}] "
                f"L{layer}H{head} "
                f"score={summary['entrainment_score']:+.6f} "
                f"clean_delta={summary['clean_gap_change']:+.6f} "
                f"distracted_delta="
                f"{summary['distracted_gap_change']:+.6f}",
                flush=True,
            )

    results.sort(
        key=lambda x: x["entrainment_score"],
        reverse=True,
    )

    json_path = output_dir / f"{split}_head_scores.json"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    csv_path = output_dir / f"{split}_head_scores.csv"

    if results:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(results[0].keys()),
            )
            writer.writeheader()
            writer.writerows(results)

    print(f"\n===== TOP 20: {split.upper()} =====")

    for rank, row in enumerate(results[:20], start=1):
        print(
            f"{rank:02d}. {row['head_name']:>6} "
            f"score={row['entrainment_score']:+.6f} "
            f"clean_delta={row['clean_gap_change']:+.6f} "
            f"distracted_delta="
            f"{row['distracted_gap_change']:+.6f} "
            f"clean_acc="
            f"{row['clean_accuracy_after_ablation']:.3f} "
            f"dist_acc="
            f"{row['distracted_accuracy_after_ablation']:.3f} "
            f"hard_rate="
            f"{row['hard_target_rate_after_ablation']:.3f}"
        )

    return {
        "baseline_entrainment_effect": baseline_effect,
        "scores": results,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--candidate-mask",
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
        "--splits",
        nargs="+",
        choices=["train", "dev", "test"],
        default=["train", "dev", "test"],
    )

    parser.add_argument(
        "--max-heads",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _, candidates = load_candidates(
        args.candidate_mask
    )

    if args.max_heads is not None:
        candidates = candidates[:args.max_heads]

    print(
        f"Candidate heads: {len(candidates)}",
        flush=True,
    )

    paper_head.HEAD_BATCH_SIZE = 1
    paper_head.set_public_seed(args.seed)

    model = paper_head.load_public_head_model(
        seed=args.seed
    )

    relation = SimpleNamespace(
        key="medqa_hard"
    )

    splits = medqa.prepare_medqa_head_experiment(
        model,
        relation,
        seed=args.seed,
        data_dir=args.data_dir,
    )

    model._unembed_last_position_only = True

    print(
        "Split counts:",
        {
            name: len(items)
            for name, items in splits.items()
        },
        flush=True,
    )

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

    report = {
        "candidate_heads": len(candidates),
        "candidate_source": str(args.candidate_mask),
        "scoring_definition": (
            "E_all - E_single_head_ablation, where "
            "E=(hard-gold)_distracted-(hard-gold)_clean"
        ),
        "splits": {},
    }

    for split in args.splits:
        split_result = evaluate_split(
            model,
            split,
            candidates,
            answer_ids,
            args.output_dir,
        )

        report["splits"][split] = {
            "baseline_entrainment_effect":
                split_result["baseline_entrainment_effect"],
            "top20": split_result["scores"][:20],
        }

    (args.output_dir / "summary.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\n===== ENTRAINMENT HEAD SCAN COMPLETE =====",
        flush=True,
    )


if __name__ == "__main__":
    main()
