"""Run configurable source/receiver blocking during free generation."""
import argparse
import csv
from pathlib import Path
from robust_medical_qa.io import read_json, write_json, new_output_dir, require, digest
from robust_medical_qa.runtime import add_runtime_arguments, load_runtime, label_tokens, ModelShape, forward
from robust_medical_qa.attention import receivers, receiver_edges
from robust_medical_qa.generation import generate
from robust_medical_qa.evaluation import outcome_counts


def run(model, tokenizer, payload, config, prompt_config, output):
    import torch
    out = new_output_dir(output)
    require(bool(payload['rows']) and 'baseline' in config['conditions'], 'Require input rows and a baseline condition')
    require(len({r['question_id'] for r in payload['rows']}) == len(payload['rows']), 'Duplicate questions')
    _, labels = label_tokens(tokenizer, prompt_config)
    layers = list(range(ModelShape.from_model(model).layers)) if config['layers'] == 'all' else config['layers']
    generation = payload['generation']
    require(generation.get('use_cache', False) is False, 'Full-prefix runtime requires use_cache=false')
    outcomes = []
    for index, row in enumerate(payload['rows']):
        for condition, specification in config['conditions'].items():
            source, policy = row['sources'][specification['source']], specification['policy']
            def step(ids):
                with torch.no_grad():
                    if policy == 'baseline':
                        return forward(model, ids).logits[0, -1].float()
                    query = receivers(policy, len(ids), source, row['options'])
                    with receiver_edges(model, layers, query, source, config['verify_attention']):
                        return forward(model, ids).logits[0, -1].float()
            record = generate(step, tokenizer, row['prompt_ids'], row, generation, labels)
            # Numeric filenames keep arbitrary question IDs out of filesystem paths.
            record_path = out / 'conditions' / f'{index:06d}_{len(outcomes) % len(config["conditions"]):02d}.json'
            write_json(record_path, {**record, 'question_id': row['question_id'], 'condition': condition})
            outcomes.append(dict(question_id=row['question_id'], condition=condition,
                role=record['parsed']['scientific_role'], answer_valid=int(record['parsed']['answer_valid']),
                finish_reason=record['finish_reason'], generated_tokens=len(record['generated_token_ids'])))
        print(f'{index+1}/{len(payload["rows"])} generated', flush=True)
    with (out / 'outcomes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(outcomes[0]))
        writer.writeheader()
        writer.writerows(outcomes)
    write_json(out / 'summary.json', outcome_counts(outcomes))
    write_json(out / 'RUN_INFO.json', {'config': config, 'generation': generation, 'label_ids': labels,
        'model_shape': ModelShape.from_model(model).__dict__, 'model_path': str(model.config._name_or_path),
        'device': str(model.device), 'dtype': str(model.dtype)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--config', default='configs/attention_blocking.json')
    parser.add_argument('--prompt-config', default='configs/prompt.json')
    parser.add_argument('--output', required=True)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    model, tokenizer = load_runtime(args.model, args.tokenizer, args.device, args.dtype)
    run(model, tokenizer, read_json(args.inputs), read_json(args.config), read_json(args.prompt_config), args.output)
    write_json(Path(args.output) / 'input_manifest.json', {'sha256': digest(args.inputs)})


if __name__ == '__main__':
    main()
