"""Generate, validate, score and select Bystander or Nonliteral distractors."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from dataclasses import asdict
import subprocess
from common.checkpoints import Run, atomic, jsonl, read, sha, sample_by_hash, child_environment
from common.hard_data.runtime import read_jsonl, insert_before_final_sentence
from common.hard_data.retry import RetryPolicy, run_target, build_scoring_pool
from common.hard_data.judge import JUDGE_PROMPT, target_gate_result
from common.hard_data.prompts import prompt_gen_beta_confounder, NONLITERAL_PROMPT
from common.hard_data.scoring import DEFAULT_MODEL, DEFAULT_REVISION
from common.hard_data.dataset import select_dataset, paired_rows, validate_scores
from common.io import hard_dataset_filename


def sources_from(rows, split):
    result = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not {'question', 'options', 'answer_idx'} <= row.keys():
            raise ValueError(f'Source row {i} needs question, options, and answer_idx')
        if not isinstance(row['question'], str) or not row['question'].strip():
            raise ValueError(f'Source row {i} needs a nonempty question')
        options = row['options']
        if (not isinstance(options, dict) or set(options) != set('ABCD')
                or any(not isinstance(v, str) or not v.strip() for v in options.values())
                or row['answer_idx'] not in ('A', 'B', 'C', 'D')):
            raise ValueError(f'Source row {i} needs nonempty A-D option strings and a single A-D answer_idx')
        idx = int(row.get('idx', i))
        sid = (f'test-{idx:05d}' if split == 'test' else str(row.get('source_id', f'{split}-{idx:05d}')))
        result.append(dict(row, idx=idx, source_id=sid))
    if not result: raise ValueError('Empty source data')
    if len({r['source_id'] for r in result}) != len(result): raise ValueError('Duplicate source IDs')
    if len({r['idx'] for r in result}) != len(result): raise ValueError('Duplicate source indices')
    if len({sha([r['question'], r['options'], r['answer_idx']]) for r in result}) != len(result):
        raise ValueError('Duplicate source questions')
    return result


def settings(kind, split, retries=2):
    if kind not in ('bystander', 'nonliteral'): raise ValueError('Unknown distractor type')
    prefix = 'nl' if kind == 'nonliteral' else 'bs'
    return dict(kind=kind, split=split, policy=asdict(RetryPolicy(extra_rounds=retries, namespace=prefix+'-cue-retry-v2')),
        generation_template=NONLITERAL_PROMPT if kind == 'nonliteral' else prompt_gen_beta_confounder,
        judge_template=JUDGE_PROMPT.replace('bystander',kind).replace('Bystander',kind.title()),
        model=DEFAULT_MODEL, revision=DEFAULT_REVISION, chat_date='2026-09-12',
        namespace=f'{prefix}-{split}-v1', random_salt=prefix+'-random-valid-v1',
        same_target_salt=prefix+'-random-same-target-v1')


def generate_pool(sources, config, directory, backend):
    directory = Path(directory)
    # Keep the original source order and target order. Only rejected slots retry.
    results = []
    policy = RetryPolicy(**config['policy'])
    for i, source in enumerate(sources):
        for target in 'ABCD':
            if target == source['answer_idx']: continue
            results.append(run_target(source, target, config['generation_template'], config['judge_template'],
                policy, directory/'candidate_checkpoints', backend.generate, backend.judge,
                target_gate_result, dict(model=config['model'], revision=config['revision'], chat_date=config['chat_date'])))
        print(f"Candidates: {i+1}/{len(sources)}", flush=True)
    pool = build_scoring_pool(sources, results, insert_before_final_sentence, sample_by_hash,
                             condition=config['kind'], random_salt=config['random_salt'])
    jsonl(directory/'target_results.jsonl', results)
    atomic(directory/'pool.json', pool)
    jsonl(directory/'score_requests.jsonl', pool['score_requests'])
    return pool


def finish(sources, config, state, output):
    pool = read(state/'pool.json')
    views, audit = select_dataset(sources, pool, read_jsonl(state/'scores.jsonl'),
        namespace=config['namespace'], random_salt=config['same_target_salt'])
    pairs = paired_rows(sources, views, audit)
    files = []
    jsonl(state/'selection_audit.jsonl',audit)
    selection=[{k:v for k,v in row.items() if not k.endswith('_hash')} for row in audit]
    for name, rows in [(hard_dataset_filename(config['kind']),pairs), ('views.jsonl',views), ('selection.jsonl',selection)]:
        jsonl(output/name,rows); files.append(output/name)
    atomic(output/'summary.json', dict(type=config['kind'], sources=len(sources), included=len(pairs),
        excluded=len(sources)-len(pairs), extra_retry_rounds=config['policy']['extra_rounds'],
        generated_candidates=sum(r['generated_n'] for r in read_jsonl(state/'target_results.jsonl')),
        accepted_candidates=len(pool['valid_candidates'])))
    files.append(output/'summary.json')
    if not pairs: raise ValueError('No eligible sources; selection audit saved')
    return files


def run(args):
    sources = sources_from(read_jsonl(args.input), args.split)
    config = settings(args.type,args.split,args.retry_rounds)
    with Run(args.output, dict(input=sha(Path(args.input)), config=config), args.resume) as run:
        state, output = run.state, run.output
        atomic(state/'sources.json', sources); atomic(state/'config.json',config)
        def candidate_stage():
            subprocess.run([getattr(args,'generation_python',sys.executable),str(Path(__file__).resolve()),'--worker',str(state)],env=child_environment(),check=True)
            return [state/'target_results.jsonl',state/'pool.json',state/'score_requests.jsonl']
        run.stage('candidates',candidate_stage)
        def scoring_stage():
            env=child_environment(CUBLAS_WORKSPACE_CONFIG=':4096:8')
            subprocess.run([getattr(args,'scoring_python',sys.executable),'-m','common.hard_data.scoring',
                '--requests',str(state/'score_requests.jsonl'),'--output',str(state/'scores.jsonl'),
                '--manifest',str(state/'score_manifest.json')],env=env,check=True)
            validate_scores(read(state/'pool.json')['score_requests'],read_jsonl(state/'scores.jsonl'))
            return [state/'scores.jsonl',state/'score_manifest.json']
        run.stage('scoring',scoring_stage)
        def repeat_stage():
            # Independent process, same first 16 requests and exact A-D logprobs.
            requests=read_jsonl(state/'score_requests.jsonl')[:16]
            jsonl(state/'repeat_requests.jsonl',requests)
            env=child_environment(CUBLAS_WORKSPACE_CONFIG=':4096:8')
            subprocess.run([getattr(args,'scoring_python',sys.executable),'-m','common.hard_data.scoring',
                '--requests',str(state/'repeat_requests.jsonl'),'--output',str(state/'repeat_scores.jsonl'),
                '--manifest',str(state/'repeat_manifest.json')],env=env,check=True)
            repeated=validate_scores(requests,read_jsonl(state/'repeat_scores.jsonl'))
            baseline={r['request_id']:r for r in read_jsonl(state/'scores.jsonl')}
            if any(r['option_logprobs']!=baseline[k]['option_logprobs'] for k,r in repeated.items()):
                raise ValueError('Independent scoring repeat did not match')
            return [state/'repeat_requests.jsonl',state/'repeat_scores.jsonl',state/'repeat_manifest.json']
        run.stage('repeat_scoring',repeat_stage)
        run.stage('selection',lambda:finish(sources,config,state,output))
        print(f"Dataset complete: {read(output/'summary.json')['included']} questions -> {output/hard_dataset_filename(args.type)}")


def main():
    parser=argparse.ArgumentParser(description='Generate, validate, and score Bystander/Nonliteral candidates and select Hard examples')
    parser.add_argument('--input',required=True,help='Clean MedQA JSONL')
    parser.add_argument('--type',required=True,choices=['bystander','nonliteral'])
    parser.add_argument('--split',choices=['dev','internal','test'],default='internal')
    parser.add_argument('--retry-rounds',type=int,choices=[0,1,2],default=2)
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--generation-python',default=sys.executable,help='Python with vLLM')
    parser.add_argument('--scoring-python',default=sys.executable,help='Python with native MXFP4 Transformers support')
    run(parser.parse_args())


if __name__ == '__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        from common.hard_data.backend import VllmRetryBackend
        state=Path(sys.argv[2]);config=read(state/'config.json')
        # Backend initializes lazily so a checkpoint-only replay needs no GPU.
        class LazyBackend:
            instance=None
            def call(self,name,*args):
                if self.instance is None: self.instance=VllmRetryBackend(config['chat_date'])
                return getattr(self.instance,name)(*args)
            def generate(self,*args): return self.call('generate',*args)
            def judge(self,*args): return self.call('judge',*args)
        generate_pool(read(state/'sources.json'),config,state,LazyBackend())
    else: main()
