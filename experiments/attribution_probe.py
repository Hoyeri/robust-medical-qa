"""Fit paired attribution probes to saved representations."""
import argparse
from dataclasses import asdict
from pathlib import Path
from robust_medical_qa.io import read_json, read_jsonl, write_json, digest, require
from robust_medical_qa.probes import ProbeConfig, load_paired, cross_validate, holdout
from robust_medical_qa.spans import attribution_metadata


def run(hidden_dir, config, items=None):
    cfg = ProbeConfig(**config['probe'])
    ids, A, B = load_paired(hidden_dir, config['variants'], config['indices'])
    metadata = attribution_metadata(ids, items) if items is not None else None
    rows = []
    for stage, indices in config['indices'].items():
        for index in indices:
            require(0 <= index < A[stage].shape[1], f'Index out of bounds: {stage}/{index}')
            a, b = A[stage][:, index], B[stage][:, index]
            result = cross_validate(a, b, cfg)
            row = {'position': stage, 'layer': index, **result}
            if metadata is not None:
                row['concept_holdout'] = cross_validate(a, b, cfg, metadata['concept'])
                row['template_holdouts'] = {family: holdout(a, b, metadata['family'], family, cfg)
                                           for family in dict.fromkeys(metadata['family'])}
            rows.append(row)
    return {'independent_pairs': len(ids), 'probe': asdict(cfg), 'variants': config['variants'],
            'source_sha256': {v: digest(Path(hidden_dir) / f'{v}.npz') for v in config['variants']}, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hidden-dir', required=True)
    parser.add_argument('--config', default='configs/attribution_probe.json')
    parser.add_argument('--items', help='JSONL metadata for concept/template holdouts')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require(not Path(args.output).exists(), 'Choose a new output file')
    result = run(args.hidden_dir, read_json(args.config), read_jsonl(args.items) if args.items else None)
    write_json(args.output, result)
    print(f"Wrote {len(result['rows'])} probe rows for {result['independent_pairs']} pairs")


if __name__ == '__main__':
    main()
