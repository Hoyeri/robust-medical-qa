#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.hard_data.runtime import (  # noqa: E402
    CHOICES,
    read_json,
    read_jsonl,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)


from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)


class CandidateDecision:
    status: str
    selected_candidate_id: str | None
    selected_by_flip_priority: bool | None
    tied_candidate_ids: tuple[str, ...]


def unique_top1(option_logprobs: dict[str, float]) -> str | None:
    maximum = max(option_logprobs.values())
    winners = [choice for choice in CHOICES if option_logprobs[choice] == maximum]
    return winners[0] if len(winners) == 1 else None


def overall_difficulty(option_logprobs: dict[str, float], gold: str) -> float:
    return max(option_logprobs[choice] for choice in CHOICES if choice != gold) - option_logprobs[gold]


def enrich_candidate(
    candidate: dict[str, Any], clean_top1: str | None, gold: str
) -> dict[str, Any]:
    logprobs = candidate["option_logprobs"]
    candidate_top1 = unique_top1(logprobs)
    difficulty = overall_difficulty(logprobs, gold)
    harmful_flip = clean_top1 == gold and candidate_top1 in CHOICES and candidate_top1 != gold
    return {
        **candidate,
        "candidate_top1": candidate_top1,
        "overall_difficulty": difficulty,
        "construction_harmful_flip": harmful_flip,
    }


def decide_candidate(
    candidates: list[dict[str, Any]], clean_top1: str | None, gold: str
) -> tuple[CandidateDecision, list[dict[str, Any]]]:
    if not candidates:
        return (
            CandidateDecision("no_gate_valid_candidate", None, None, ()),
            [],
        )
    enriched = [enrich_candidate(candidate, clean_top1, gold) for candidate in candidates]
    flip_candidates = [row for row in enriched if row["construction_harmful_flip"]]
    pool = flip_candidates if flip_candidates else enriched
    maximum = max(row["overall_difficulty"] for row in pool)
    winners = [row for row in pool if row["overall_difficulty"] == maximum]
    winners.sort(key=lambda row: row["candidate_id"])
    if len(winners) != 1:
        return (
            CandidateDecision(
                "unresolved_exact_D_tie",
                None,
                bool(flip_candidates),
                tuple(row["candidate_id"] for row in winners),
            ),
            enriched,
        )
    winner = winners[0]
    status = (
        "selected_from_flip_candidates"
        if flip_candidates
        else "selected_by_max_D_fallback"
    )
    return (
        CandidateDecision(
            status,
            winner["candidate_id"],
            bool(flip_candidates),
            (),
        ),
        enriched,
    )

RESIDUAL_HASH_SALT = 'gpt-oss-hard-v2-residual-tie-v1'


@dataclass(frozen=True)


class ResidualCandidateDecision:
    final_status: str
    selected_candidate_id: str | None
    selected_by_flip_priority: bool | None
    base_selection_status: str
    resolution_method: str
    tied_candidate_ids: tuple[str, ...]
    canonical_candidate_ids: tuple[str, ...]
    discarded_duplicate_candidate_ids: tuple[str, ...]
    selected_tie_break_hash: str | None


def deterministic_tie_hash(
    candidate: dict[str, Any],
    source_id: str,
    *,
    salt: str = RESIDUAL_HASH_SALT,
) -> str:
    fields = (
        salt,
        source_id,
        str(candidate["candidate_id"]),
        str(candidate["candidate_text_sha256"]),
        str(candidate["intended_target"]),
        str(candidate["target_gate_label"]),
    )
    return sha256("|".join(fields).encode("utf-8")).hexdigest()


def _duplicate_identity(candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(candidate["candidate_text_sha256"]),
        str(candidate["intended_target"]),
        str(candidate["target_gate_label"]),
    )


def _canonical_duplicate_representatives(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        for required in (
            "candidate_id",
            "candidate_rank",
            "candidate_text_sha256",
            "intended_target",
            "target_gate_label",
        ):
            if required not in candidate:
                raise KeyError(f"residual selector requires {required}")
        groups.setdefault(_duplicate_identity(candidate), []).append(candidate)

    representatives: list[dict[str, Any]] = []
    discarded: list[str] = []
    for rows in groups.values():
        ordered = sorted(
            rows,
            key=lambda row: (int(row["candidate_rank"]), str(row["candidate_id"])),
        )
        representatives.append(ordered[0])
        discarded.extend(str(row["candidate_id"]) for row in ordered[1:])
    representatives.sort(key=lambda row: str(row["candidate_id"]))
    discarded.sort()
    return representatives, discarded


def decide_candidate_with_residual_policy(
    candidates: list[dict[str, Any]],
    clean_top1: str | None,
    gold: str,
    source_id: str,
) -> tuple[ResidualCandidateDecision, list[dict[str, Any]]]:
    base, enriched = decide_candidate(candidates, clean_top1, gold)

    if base.selected_candidate_id is not None:
        return (
            ResidualCandidateDecision(
                final_status="selected",
                selected_candidate_id=base.selected_candidate_id,
                selected_by_flip_priority=base.selected_by_flip_priority,
                base_selection_status=base.status,
                resolution_method="primary_rule_unique_winner",
                tied_candidate_ids=(),
                canonical_candidate_ids=(),
                discarded_duplicate_candidate_ids=(),
                selected_tie_break_hash=None,
            ),
            enriched,
        )

    if base.status == "no_gate_valid_candidate":
        return (
            ResidualCandidateDecision(
                final_status="excluded_construction_infeasible",
                selected_candidate_id=None,
                selected_by_flip_priority=None,
                base_selection_status=base.status,
                resolution_method="no_gate_valid_exclusion",
                tied_candidate_ids=(),
                canonical_candidate_ids=(),
                discarded_duplicate_candidate_ids=(),
                selected_tie_break_hash=None,
            ),
            enriched,
        )

    if base.status != "unresolved_exact_D_tie":
        raise ValueError(f"unsupported residual selector state: {base.status}")

    enriched_index = {str(row["candidate_id"]): row for row in enriched}
    tied = [enriched_index[candidate_id] for candidate_id in base.tied_candidate_ids]
    representatives, discarded = _canonical_duplicate_representatives(tied)
    canonical_ids = tuple(str(row["candidate_id"]) for row in representatives)

    if len(representatives) == 1:
        winner = representatives[0]
        method = "duplicate_text_canonical_provenance"
        selected_hash = None
    else:
        hashed = [
            (deterministic_tie_hash(row, source_id), str(row["candidate_id"]), row)
            for row in representatives
        ]
        selected_hash, _, winner = min(hashed, key=lambda value: (value[0], value[1]))
        method = "distinct_candidate_deterministic_hash"

    return (
        ResidualCandidateDecision(
            final_status="selected",
            selected_candidate_id=str(winner["candidate_id"]),
            selected_by_flip_priority=base.selected_by_flip_priority,
            base_selection_status=base.status,
            resolution_method=method,
            tied_candidate_ids=base.tied_candidate_ids,
            canonical_candidate_ids=canonical_ids,
            discarded_duplicate_candidate_ids=tuple(discarded),
            selected_tie_break_hash=selected_hash,
        ),
        enriched,
    )
