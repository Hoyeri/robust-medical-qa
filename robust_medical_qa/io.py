"""File IO and experiment metadata."""
import hashlib
import json
import os
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    require(bool(rows), 'Input is empty')
    return rows


def write_json(path, value, overwrite=False):
    path = Path(path)
    require(overwrite or not path.exists(), f'Output already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def new_output_dir(path):
    path = Path(path)
    require(not path.exists() or not any(path.iterdir()), f'Output directory is not empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path
