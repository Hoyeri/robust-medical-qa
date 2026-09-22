from __future__ import annotations
import hashlib
from typing import Any, Iterable, Mapping
from .core import validate_source_pair
from .parsing import development_candidate_to_pair
CALIBRATION_SALT = "gpt-oss-only-hard-v2-external-eval-v3-split-tracks-calibration-v1"

def select_calibration_pairs(
    rows: Iterable[Mapping[str, Any]], *, per_stratum: int = 16
) -> list[dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = {
        "harmful_flip_priority": [],
        "maximum_D_fallback": [],
    }
    for raw in rows:
        pair = dict(raw) if 'distracted_question' in raw else development_candidate_to_pair(raw)
        validate_source_pair(pair)
        pools[pair["selection_pool"]].append(pair)
    selected: list[dict[str, Any]] = []
    for stratum in ("harmful_flip_priority", "maximum_D_fallback"):
        ranked = sorted(
            pools[stratum],
            key=lambda row: hashlib.sha256(
                f"{CALIBRATION_SALT}:{row['question_id']}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < per_stratum:
            raise ValueError(f"insufficient calibration rows in {stratum}")
        selected.extend(ranked[:per_stratum])
    return sorted(selected, key=lambda row: (row["selection_pool"], row["question_id"]))
