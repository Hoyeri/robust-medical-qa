"""Analyze attribution decoding and diagnostic sensitivity from saved transmission arrays."""
import argparse
from pathlib import Path
from robust_medical_qa.io import read_json, read_jsonl, write_json, require
from robust_medical_qa.probes import ProbeConfig, load_paired, cross_validate, holdout
from robust_medical_qa.spans import attribution_metadata
from robust_medical_qa.evaluation import paired_summary


def analyze(directory, items, config):
    cfg = ProbeConfig(**config['probe'])
    ids, A, B = load_paired(directory, config['variants'], config['indices'])
    meta = attribution_metadata(ids, items)
    decoded, summaries = [], {}
    for stage, indices in config['indices'].items():
        minima = {}
        for index in indices:
            require(index < A[stage].shape[1], f'Index outside {stage}')
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
                        values.append(sum(x[key][l] for l in config['sensitivity_layers']) - sum(y[key][l] for l in config['sensitivity_layers']))
                if values:
                    sensitivity.append(dict(readout=key, a=a, b=b, source_a=sa, source_b=sb,
                                            **paired_summary(values, **bootstrap)))
    values = [alignment[i]['Ma']['q'] - alignment[i]['H1']['q'] for i in ids]
    answer = {s: paired_summary(values, statistic=s, **bootstrap) for s in ('mean','median')}
    validation_path = Path(directory) / 'validation.json'
    validation = read_json(validation_path) if validation_path.exists() else []
    return dict(independent_pairs=len(ids), config=config, decoding=decoded, stage_summary=summaries,
                sensitivity=sensitivity, paired_answer_score=answer, validation=validation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--items', required=True)
    parser.add_argument('--config', default='configs/transmission_analysis.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require(not Path(args.output).exists(), 'Choose a new output file')
    report = analyze(args.results_dir, read_jsonl(args.items), read_json(args.config))
    write_json(args.output, report)
    for row in report['validation']:
        print(f"{row['item_id']} layer={row['layer']} alpha={row['alpha']} actual={row['dq_actual']:.4f} first_order={row['dq_first_order']:.4f}")
    print(report['stage_summary'])


if __name__ == '__main__':
    main()
