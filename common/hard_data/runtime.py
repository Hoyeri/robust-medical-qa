from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


CHOICES = ("A", "B", "C", "D")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def canonical_jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def write_bytes(path: str | Path, payload: bytes, *, overwrite: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = destination.read_bytes()
        if existing == payload:
            return
        if not overwrite:
            raise RuntimeError(
                f"refusing to overwrite a different artifact without --overwrite: {destination}"
            )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def write_json(
    path: str | Path, value: Any, *, overwrite: bool = False
) -> None:
    write_bytes(path, canonical_json_bytes(value), overwrite=overwrite)


def write_jsonl(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    overwrite: bool = False,
) -> None:
    write_bytes(path, canonical_jsonl_bytes(rows), overwrite=overwrite)


def write_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    *,
    overwrite: bool = False,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = temporary.read_bytes()
    temporary.unlink()
    write_bytes(destination, payload, overwrite=overwrite)


def unique_index(
    rows: Iterable[dict[str, Any]], key: str, label: str
) -> dict[Any, dict[str, Any]]:
    output: dict[Any, dict[str, Any]] = {}
    for row in rows:
        value = row[key]
        if value in output:
            raise ValueError(f"duplicate {label}: {value!r}")
        output[value] = row
    return output


def insert_before_final_sentence(question: str, distractor: str) -> str:
    """Exact MedDistractQA insertion parity used by the legacy pipeline."""
    import re

    sentences = re.split(r"(?<=[.!?]) +", question.strip())
    sentences.insert(-1, distractor.strip())
    return " ".join(sentences)
