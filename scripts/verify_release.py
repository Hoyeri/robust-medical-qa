"""Verify release bytes, original-code snapshot hashes and result arithmetic."""
import hashlib
import json
from pathlib import Path
from reproduce_tables import check_tables

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / 'provenance/release_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert digest(ROOT / name) == expected, f'Changed or missing release file: {name}'
    sources = json.loads((ROOT / 'provenance/sources.json').read_text())
    copies = [r for r in sources['sources'] if 'included_copy' in r]
    for r in copies:
        assert digest(ROOT / r['included_copy']) == r.get('included_sha256', r['sha256']), r['included_copy']
    check_tables()
    print(f"PASS: {len(manifest['files'])} release files, {len(copies)} source-code snapshots, result counts.")
    print('This checks the sharing snapshot; it does not rerun GPU inference.')


if __name__ == '__main__':
    main()
