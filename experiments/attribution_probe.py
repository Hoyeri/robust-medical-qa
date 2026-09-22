"""Fit paired attribution probes to saved representations."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from dataclasses import asdict
from common.checkpoints import Run, atomic, sha
from common.io import read_json, read_jsonl, digest, require
from common.runtime import add_runtime_arguments, load_runtime, model_binding
from common.extraction import extract
from common.probes import ProbeConfig, load_paired, cross_validate, holdout
from common.spans import attribution_metadata


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
    parser.add_argument('--input', default=ROOT/'common/attribution_data.jsonl', help='Attribution JSONL; defaults to the included 118-item dataset')
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', help='Optional experiment configuration JSON')
    parser.add_argument('--resume', action='store_true')
    add_runtime_arguments(parser)
    args = parser.parse_args()
    cfg = read_json(args.config or ROOT / 'common/configs/attribution_probe.json')
    prompt = read_json(ROOT / 'common/prompt.json')
    binding = dict(input=sha(Path(args.input)), config=cfg, prompt=prompt,
                   model=model_binding(args.model, args.tokenizer), device=args.device, dtype=args.dtype)
    with Run(args.output, binding, args.resume) as job:
        arrays = job.state / 'representations'

        def compute():
            model, tok = load_runtime(args.model, args.tokenizer, args.device, args.dtype)
            extract(model, tok, args.input, arrays, cfg['extraction'], prompt, 'hidden', cfg.get('validate', 0), resume=True)
            return list(arrays.glob('*.npz')) + list(arrays.glob('*.json'))

        job.stage('extraction', compute)

        def analysis():
            report = {'by_item': run(arrays, cfg['analysis']),
                      'holdouts': run(arrays, cfg['holdouts'], read_jsonl(args.input))}
            for result in report.values():
                result.pop('source_sha256', None)
            atomic(job.output / 'results.json', report)
            return [job.output / 'results.json']

        job.stage('analysis', analysis)
        print('Complete: ' + str(job.output))


if __name__ == "__main__":
    main()
