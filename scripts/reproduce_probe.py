"""Recompute and compare the frozen attribution-probe results."""
import argparse
import os
import sys
from pathlib import Path
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(variable, '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.attribution_probe import run
from robust_medical_qa.io import read_json, write_json, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hidden-dir', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', default=str(ROOT / 'configs/attribution_probe.json'))
    parser.add_argument('--expected', default=str(ROOT / 'results/role_probe_exact.json'))
    args = parser.parse_args()
    require(not args.output.exists(), 'Choose a new output filename')
    result = run(args.hidden_dir, read_json(args.config))
    expected = read_json(args.expected)
    reference = {(r['position'], r['layer']): r for r in expected['rows']}
    require(len(reference) == len(result['rows']), 'Probe row inventory differs')
    matched = result['independent_pairs'] == expected['independent_items']
    for row in result['rows']:
        counts = [r['correct'] for r in row['repeats']]
        target = reference[(row['position'], row['layer'])]
        row['matches_frozen_counts'] = counts == target['correct_per_seed'] and all(r['n'] == target['denominator_per_seed'] for r in row['repeats'])
        matched &= row['matches_frozen_counts']
        print(f"{row['position']} index {row['layer']}: {counts}; match={row['matches_frozen_counts']}", flush=True)
    write_json(args.output, result)
    require(matched, 'Counts differ; inspect input hashes and numerical environment')
    print(f"PASS: {len(result['rows'])} probe rows match the frozen counts")


if __name__ == '__main__':
    main()
