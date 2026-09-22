"""Run configurable source/receiver blocking during free generation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import csv
from common.checkpoints import Run, Checkpoints, atomic, sha
from common.io import read_json, read_jsonl, new_output_dir, require, digest, hard_dataset_path
from common.runtime import add_runtime_arguments, load_runtime, model_binding, label_tokens, ModelShape, forward
from common.attention import receivers, receiver_edges
from common.generation import generate
from common.evaluation import outcome_counts


def run(model, tokenizer, payload, config, prompt_config, output, resume=False):
    import torch
    out = Path(output) if resume else new_output_dir(output)
    out.mkdir(parents=True,exist_ok=True)
    cache = Checkpoints(out/'.state')
    require(bool(payload['rows']) and 'baseline' in config['conditions'], 'Require input rows and a baseline condition')
    require(config['conditions']['baseline']['policy'] == 'baseline', 'Baseline condition must use baseline policy')
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
            record = cache.cached(f'{index}:{condition}', dict(row=row, config=config, generation=generation, labels=labels),
                lambda: generate(step, tokenizer, row['prompt_ids'], row, generation, labels))
            # Numeric filenames keep arbitrary question IDs out of filesystem paths.
            record_path = out / 'conditions' / f'{index:06d}_{len(outcomes) % len(config["conditions"]):02d}.json'
            atomic(record_path, {**record, 'question_id': row['question_id'], 'condition': condition})
            outcomes.append(dict(question_id=row['question_id'], condition=condition,
                role=record['parsed']['scientific_role'], answer_valid=int(record['parsed']['answer_valid']),
                finish_reason=record['finish_reason'], generated_tokens=len(record['generated_token_ids'])))
        print(f'{index+1}/{len(payload["rows"])} generated', flush=True)
    with (out / 'outcomes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(outcomes[0]))
        writer.writeheader()
        writer.writerows(outcomes)
    atomic(out / 'summary.json', outcome_counts(outcomes))
    atomic(out / 'RUN_INFO.json', {'config': config, 'generation': generation, 'label_ids': labels,
        'model_shape': ModelShape.from_model(model).__dict__, 'model_path': str(model.config._name_or_path),
        'device': str(model.device), 'dtype': str(model.dtype)})


def convert(distractor_input, clinical_control_input):
    base = read_json(distractor_input)
    control = read_json(clinical_control_input)
    require(base['cfg'] == control['cfg'], 'Generation configs differ')
    lookup = {r['question_id']: r for r in control['rows']}
    ids = [r['question_id'] for r in base['rows']]
    require(len(ids) == len(set(ids)) and set(ids) == set(lookup) and len(lookup) == len(control['rows']), 'Question inventories differ')
    rows = []
    for row in base['rows']:
        other = lookup[row['question_id']]
        require(row['gold_answer'] == other['gold_answer'] and row['intended_target'] == other['intended_target'], 'Answer labels differ')
        require(row['prompt_ids'] == other['prompt_ids'] and row['options'] == other['options'], 'Prompt/options differ')
        require(len(row['source']) == len(other['source']) and not set(row['source']) & set(other['source']), 'Control source mismatch')
        rows.append({k: row[k] for k in ('question_id', 'gold_answer', 'intended_target', 'prompt_ids', 'options')} |
                    {'sources': {'distractor': row['source'], 'clinical': other['source']}})
    return {'generation': base['cfg'], 'rows': rows, 'source_sha256': {'distractor_input': digest(distractor_input), 'clinical_control_input': digest(clinical_control_input)}}


def from_pairs(pairs, tokenizer, prompt):
    import hashlib
    from common.runtime import prompt_tokens
    from common.spans import unique_span, overlapping
    if not pairs: raise ValueError('Empty pairs')
    rows=[]
    for pair in pairs:
        question=pair['distracted_question']; sid=pair['question_id']
        ids,options,text=prompt_tokens(tokenizer,question,pair['options'],prompt)
        offsets=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
        span=unique_span(text,question)
        a,b=unique_span(question,pair['added_distractor'])
        distractor=overlapping(offsets,(span[0]+a,span[0]+b))
        clinical=[i for i in overlapping(offsets,span) if i not in set(distractor)]
        n=len(distractor)
        require(n>0 and len(clinical)>=n,'Insufficient clinical tokens for exact length control')
        start=int(hashlib.sha256(sid.encode()).hexdigest()[:16],16)%(len(clinical)-n+1)
        matched=clinical[start:start+n]
        rows.append(dict(question_id=sid,gold_answer=pair['gold_answer'],intended_target=pair['intended_target'],
                         prompt_ids=ids,options=options,sources=dict(distractor=distractor,clinical=matched)))
    # Llama 3.1 generation stop IDs in the original experiment.
    stops=[tokenizer.convert_tokens_to_ids(t) for t in ('<|end_of_text|>','<|eom_id|>','<|eot_id|>')]
    require(all(isinstance(t,int) and t!=tokenizer.unk_token_id for t in stops),'Missing Llama stop tokens')
    return dict(rows=rows,generation=dict(max_new_tokens=1024,use_cache=False,stop_token_ids=stops,
                                         assistant_prefill='',json_constrained_decoding=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Hard dataset JSONL or generation output directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', help='Optional experiment configuration JSON')
    parser.add_argument('--resume', action='store_true')
    add_runtime_arguments(parser)
    args = parser.parse_args()
    input_path = hard_dataset_path(args.input)
    cfg = read_json(args.config or ROOT / 'common/configs/attention_blocking.json')
    prompt = read_json(ROOT / 'common/prompt.json')
    binding = dict(input=sha(input_path), config=cfg, prompt=prompt,
                   model=model_binding(args.model, args.tokenizer), device=args.device, dtype=args.dtype)
    with Run(args.output, binding, args.resume) as job:
        def compute():
            model, tok = load_runtime(args.model, args.tokenizer, args.device, args.dtype)
            payload = (read_json(input_path) if input_path.suffix == '.json'
                       else from_pairs(read_jsonl(input_path), tok, prompt))
            destination = job.state / 'generation'
            run(model, tok, payload, cfg, prompt, destination, resume=True)
            atomic(job.output / 'summary.json', read_json(destination / 'summary.json'))
            (job.output / 'outcomes.csv').write_bytes((destination / 'outcomes.csv').read_bytes())
            return [job.output / 'summary.json', job.output / 'outcomes.csv'] + list((destination / 'conditions').glob('*.json'))

        job.stage('attention', compute)
        print('Complete: ' + str(job.output))


if __name__ == "__main__":
    main()
