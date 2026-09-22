"""Analyze attribution decoding and diagnostic sensitivity from saved transmission arrays."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from common.checkpoints import Run, atomic, sha
from common.io import read_json, read_jsonl, require
from common.runtime import add_runtime_arguments, load_runtime, model_binding
from common.extraction import extract
from common.probes import ProbeConfig, load_paired, cross_validate, holdout
from common.spans import attribution_metadata
from common.evaluation import paired_summary


def analyze(directory, items, config):
    cfg = ProbeConfig(**config['probe'])
    for name in ('summary_indices', 'sensitivity_layers'):
        indices = config[name]
        require(bool(indices) and all(type(i) is int and i >= 0 for i in indices)
                and len(indices) == len(set(indices)), f'{name} must contain distinct nonnegative indices')
    ids, A, B = load_paired(directory, config['variants'], config['indices'])
    meta = attribution_metadata(ids, items)
    decoded, summaries = [], {}
    for stage, indices in config['indices'].items():
        minima = {}
        for index in indices:
            require(type(index) is int and 0 <= index < min(A[stage].shape[1], B[stage].shape[1]),
                    f'Index outside {stage}: {index}')
            a, b = A[stage][:, index], B[stage][:, index]
            by_item = cross_validate(a, b, cfg)
            concept = cross_validate(a, b, cfg, meta['concept'])
            templates = {f: holdout(a, b, meta['family'], f, cfg) for f in dict.fromkeys(meta['family'])}
            shuffled = cross_validate(a, b, cfg, shuffle='pair_swap')
            minimum = min([concept['mean_accuracy']] + [r['accuracy'] for r in templates.values()])
            minima[index] = minimum
            decoded.append(dict(stage=stage, index=index, by_item=by_item, concept=concept,
                                templates=templates, shuffled=shuffled, minimum_holdout=minimum))
        require(set(config['summary_indices']) <= set(minima), 'Missing summary index')
        mean = sum(minima[i] for i in config['summary_indices']) / len(config['summary_indices'])
        summaries[stage] = {'mean_minimum_holdout': mean, 'indices': config['summary_indices']}
    alignment = read_json(Path(directory) / 'alignment.json')
    bootstrap = config['bootstrap']
    sensitivity = []
    for key in ('A_all', 'A_ans'):
        for source in ('sent', 'phr'):
            comparisons = [(a,b,source,source) for a,b in config['contrasts']] + [('Ma','Ma',source,'rand')]
            for a,b,sa,sb in comparisons:
                values = []
                for item in alignment.values():
                    x, y = item.get(a, {}).get('sources', {}).get(sa), item.get(b, {}).get('sources', {}).get(sb)
                    if x is not None and y is not None:
                        require(all(l < min(len(x[key]), len(y[key])) for l in config['sensitivity_layers']),
                                f'Sensitivity layer outside {key}: {a}/{sa}, {b}/{sb}')
                        values.append(sum(x[key][l] for l in config['sensitivity_layers']) - sum(y[key][l] for l in config['sensitivity_layers']))
                if values:
                    sensitivity.append(dict(readout=key, a=a, b=b, source_a=sa, source_b=sb,
                                            **paired_summary(values, **bootstrap)))
    # Every decoded pair must contribute to the answer comparison; never drop missing rows.
    for i in ids:
        require(i in alignment and all('q' in alignment[i].get(v, {}) for v in ('Ma', 'H1')),
                f'Missing paired answer score: {i}')
    values = [alignment[i]['Ma']['q'] - alignment[i]['H1']['q'] for i in ids]
    answer = {s: paired_summary(values, statistic=s, **bootstrap) for s in ('mean','median')}
    validation_path = Path(directory) / 'validation.json'
    validation = read_json(validation_path) if validation_path.exists() else []
    return dict(independent_pairs=len(ids), config=config, decoding=decoded, stage_summary=summaries,
                sensitivity=sensitivity, paired_answer_score=answer, validation=validation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=ROOT/'common/attribution_data.jsonl', help='Attribution JSONL; defaults to the included 118-item dataset')
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', help='Optional experiment configuration JSON')
    parser.add_argument('--resume', action='store_true')
    add_runtime_arguments(parser)
    args = parser.parse_args()
    cfg = read_json(args.config or ROOT / 'common/configs/transmission.json')
    prompt = read_json(ROOT / 'common/prompt.json')
    binding = dict(input=sha(Path(args.input)), config=cfg, prompt=prompt,
                   model=model_binding(args.model, args.tokenizer), device=args.device, dtype=args.dtype)
    with Run(args.output, binding, args.resume) as job:
        arrays = job.state / 'representations'

        def compute():
            model, tok = load_runtime(args.model, args.tokenizer, args.device, args.dtype)
            extract(model, tok, args.input, arrays, cfg['extraction'], prompt, 'transmission', cfg.get('validate', 0), resume=True)
            return list(arrays.glob('*.npz')) + list(arrays.glob('*.json'))

        job.stage('extraction', compute)

        def analysis():
            report = analyze(arrays, read_jsonl(args.input), cfg['analysis'])
            atomic(job.output / 'results.json', report)
            return [job.output / 'results.json']

        job.stage('analysis', analysis)
        print('Complete: ' + str(job.output))


if __name__ == "__main__":
    main()
