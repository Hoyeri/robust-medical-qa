"""File IO and experiment metadata."""
import hashlib
import json
import os
from pathlib import Path

HARD_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / 'outputs'


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


def hard_dataset_filename(family, *, full_coverage=False):
    require(family in ('bystander', 'nonliteral'), 'Unknown Hard dataset type')
    suffix = '-fullcoverage-v2' if full_coverage else ''
    return f'meddistractqa-hard-{family}{suffix}.jsonl'


def hard_dataset_filenames(family):
    return [hard_dataset_filename(family), hard_dataset_filename(family, full_coverage=True)]


def hard_dataset_path(path=None, *, family=None):
    """Resolve an explicit input or the standard construction output folders."""
    if path is None:
        families = ('bystander', 'nonliteral') if family is None else (family,)
        candidates = []
        for kind in families:
            directory = HARD_OUTPUT_ROOT / kind
            family_candidates = [directory / name for name in hard_dataset_filenames(kind)]
            family_candidates = [dataset for dataset in family_candidates if dataset.is_file()]
            if not family_candidates and (directory / 'pairs.jsonl').is_file():
                family_candidates = [directory / 'pairs.jsonl']
            candidates.extend(family_candidates)
        searched = ', '.join(str(HARD_OUTPUT_ROOT / kind) for kind in families)
        require(bool(candidates), f'No Hard dataset found in {searched}. '
                'Generate it with pipeline/build_dataset.py or provide --input PATH.')
        require(len(candidates) == 1, 'Multiple Hard datasets found: '
                + ', '.join(str(p) for p in candidates) + '. Select one with --input PATH.')
        return candidates[0]
    path = Path(path)
    if path.is_dir():
        candidates = [path / name for family in ('bystander', 'nonliteral')
                      for name in hard_dataset_filenames(family)]
        candidates = [p for p in candidates if p.is_file()]
        if not candidates and (path / 'pairs.jsonl').is_file():
            candidates = [path / 'pairs.jsonl']
        require(len(candidates) == 1, f'Expected one Hard dataset in {path}; specify a dataset file explicitly')
        return candidates[0]
    require(path.is_file(), f'Dataset file not found: {path}')
    return path
