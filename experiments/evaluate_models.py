"""Prepare, calibrate, evaluate and summarize models from paired data."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import subprocess
from common.checkpoints import Run, atomic, jsonl, read, sha, child_environment
from common.hard_data.runtime import read_jsonl, sha256_file, write_json, write_jsonl
from common.model_eval.protocol import (align_source_triads, build_full_requests, build_calibration_requests,
    select_calibration_pairs, prompt_contract_sha256, response_compatibility_sha256, _request,
    EXPECTED_PROTOCOL_STATUS, DIRECT_PROTOCOL_STATUS)
from common.model_eval.core import validate_source_pair
from common.io import hard_dataset_path

MODEL_CONFIG = ROOT / "common/model_eval/models.json"
OFFICIAL_REPO = 'KrithikV/MedDistractQA'
OFFICIAL_REVISION = '4e57408a3a1113a350674088cdcd34c770d81a25'
OFFICIAL_FILES = {'bystander': 'MedDistractQA_Bystander.json',
                  'nonliteral': 'MedDistractQA_Nonliteral.json'}


def official_source(family):
    return dict(repo_id=OFFICIAL_REPO, revision=OFFICIAL_REVISION,
                filename=OFFICIAL_FILES[family], family=family)


def match_official(rows, pairs, family):
    """Match HF records by recovered Clean text, then verify options and gold."""
    if not isinstance(rows, list) or not rows:
        raise ValueError('Official dataset must be a nonempty JSON array')
    normalize = lambda text: ' '.join(text.split())
    index = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or not {'question', 'distracting_sentence', 'question_choices', 'correct_answer'} <= row.keys():
            raise ValueError(f'Official row {position} needs question, distracting_sentence, question_choices, and correct_answer')
        question, sentence = row['question'], row['distracting_sentence']
        if not isinstance(question, str) or not isinstance(sentence, str) or not sentence.strip():
            raise ValueError(f'Invalid official question/distractor at row {position}')
        if question.count(sentence) != 1:
            raise ValueError(f'Official distractor must occur exactly once at row {position}')
        clean = normalize(question.replace(sentence, '', 1))
        if not clean or clean in index:
            raise ValueError(f'Empty or duplicate official Clean question at row {position}')
        if (not isinstance(row['question_choices'], dict) or set(row['question_choices']) != set('ABCD')
                or any(not isinstance(v, str) or not v.strip() for v in row['question_choices'].values())
                or row['correct_answer'] not in ('A', 'B', 'C', 'D')):
            raise ValueError(f'Invalid official options/gold at row {position}')
        index[clean] = (position, row)
    matched, used, mapping = [], set(), []
    condition = {'bystander': 'B0_BS_OFFICIAL', 'nonliteral': 'B0_NL_OFFICIAL'}[family]
    for pair in pairs:
        validate_source_pair(pair)
        sid = pair['question_id']
        candidate = index.get(normalize(pair['clean_question']))
        if candidate is None:
            raise ValueError(f'No official {family} match for {sid}; evaluation stopped before inference')
        position, row = candidate
        if position in used:
            raise ValueError(f'Duplicate Hard source question: {sid}')
        if row['question_choices'] != pair['options']:
            raise ValueError(f'Official options mismatch: {sid}')
        if row['correct_answer'] != pair['gold_answer']:
            raise ValueError(f'Official gold mismatch: {sid}')
        used.add(position)
        # Preserve the official input verbatim; normalize whitespace only for matching.
        matched.append(dict(idx=pair['source_idx'], source_id=sid,
            clean_question=pair['clean_question'], question=row['question'], options=row['question_choices'],
            answer_idx=row['correct_answer'], distracting_sentence=row['distracting_sentence'], condition=condition))
        mapping.append(dict(question_id=sid, official_row=position))
    metadata = dict(**official_source(family), available_questions=len(rows), matched_questions=len(matched),
                    scope='hard_input_questions', alignment=mapping)
    return matched, metadata


def load_official(pairs, family):
    from huggingface_hub import hf_hub_download
    source = official_source(family)
    path = Path(hf_hub_download(repo_id=source['repo_id'], repo_type='dataset',
        filename=source['filename'], revision=source['revision'], token=False))
    matched, metadata = match_official(read(path), pairs, family)
    metadata['file_sha256'] = sha(path)
    return matched, metadata


def result_table(summary):
    """Use the existing aggregate counts, including invalid answers in accuracy."""
    by_view = summary['results']['by_view']
    clean_accuracy = by_view['clean']['accuracy']
    table = []
    for view in ('clean', 'meddistractqa', 'hard'):
        if view not in by_view:
            continue
        counts = by_view[view]
        n, correct = counts['n'], counts['correct']['count']
        invalid = counts['role_counts'].get('Invalid', 0)
        table.append(dict(condition=view, questions=n, correct=correct, wrong=n-correct-invalid,
            invalid=invalid, accuracy=counts['accuracy'],
            accuracy_change_from_clean_pp=100*(counts['accuracy']-clean_accuracy),
            max_token_cutoff=counts['max_token_cutoff']))
    return table


def format_results(summary):
    labels = {'clean': 'Clean', 'meddistractqa': 'Official', 'hard': 'Hard'}
    lines = [f"Model: {summary['model']}", f"Paired questions: {summary['results']['questions']}"]
    if summary.get('official'):
        source = summary['official']
        lines.append(f"Official: {source['repo_id']} / {source['family']} / "
                     f"matched {source['matched_questions']} of {source['available_questions']}")
    lines += ['', 'Condition | N | Correct | Wrong | Invalid | Accuracy | Change vs Clean (pp) | Token cutoff',
              '--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:']
    for row in result_table(summary):
        lines.append(f"{labels[row['condition']]} | {row['questions']} | {row['correct']} | {row['wrong']} | "
                     f"{row['invalid']} | {row['accuracy']:.2%} | "
                     f"{row['accuracy_change_from_clean_pp']:+.2f} | {row['max_token_cutoff']}")
    lines.append('\nAccuracy includes invalid answers in the denominator. Token cutoff can overlap other columns.')
    return '\n'.join(lines) + '\n'


def paired_requests(pairs):
    requests=[]; ids=set()
    for pair in pairs:
        validate_source_pair(pair)
        if pair['question_id'] in ids: raise ValueError('Duplicate source ID')
        ids.add(pair['question_id'])
        for view in ('clean','hard'):
            requests.append(_request(request_set='full',question_id=pair['question_id'],source_idx=pair['source_idx'],
                view=view,question=pair['clean_question' if view=='clean' else 'distracted_question'],options=pair['options'],
                gold_answer=pair['gold_answer'],hard_intended_target=pair['intended_target'],
                view_distractor=None if view=='clean' else pair['added_distractor'],hard_selection_pool=pair['selection_pool'],
                hard_construction_harmful_flip=pair.get('construction_harmful_flip',False),
                hard_construction_candidate_top1_role=pair.get('construction_candidate_top1_role')))
    return requests


def prepare(pairs, development, official, registry, model, state):
    if not pairs: raise ValueError('Empty evaluation pairs')
    fp=lambda p:sha([p['clean_question'],p['options'],p['gold_answer']])
    dev_pairs=select_calibration_pairs(development) if development is not None else []
    if {fp(r) for r in pairs} & {fp(r) for r in dev_pairs}: raise ValueError('Calibration overlaps evaluation data')
    if {r['question_id'] for r in pairs} & {r['question_id'] for r in dev_pairs}: raise ValueError('Calibration source IDs overlap evaluation')
    calibration=build_calibration_requests(dev_pairs)
    full=build_full_requests(align_source_triads(pairs,official)) if official is not None else paired_requests(pairs)
    requests={'full':full}
    if development is not None:
        requests={'calibration':calibration,**requests}
    for name,rows in requests.items():
        write_jsonl(state/(name+'.jsonl'),rows,overwrite=True)
    config=dict(status=EXPECTED_PROTOCOL_STATUS if development is not None else DIRECT_PROTOCOL_STATUS,
        primary_model_keys=[model],models={model:registry['models'][model]},
        evaluation=registry['evaluation'],calibration_gate=registry['calibration_gate'],
        counts=dict(calibration_pairs=len(dev_pairs),calibration_requests_per_model=len(calibration),
                    full_requests_per_model=len(full),full_triads=len(pairs),source_items=len(pairs),construction_infeasible_exclusions=0),
        prompt_contract=dict(sha256=prompt_contract_sha256()),
        response_compatibility_contract=dict(sha256=response_compatibility_sha256()),
        bindings={name+'_requests':dict(sha256=sha256_file(state/(name+'.jsonl'))) for name in requests})
    write_json(state/'protocol.json',config,overwrite=True)
    return [state/'protocol.json']+[state/(name+'.jsonl') for name in requests]


def run(args):
    registry=read(MODEL_CONFIG)
    if args.model not in registry['primary_model_keys']: raise ValueError('Unknown evaluation model')
    input_path=hard_dataset_path(args.input)
    calibration_path=hard_dataset_path(args.calibration) if getattr(args,'calibration',None) else None
    paths={'input':input_path}
    if calibration_path is not None: paths['calibration']=calibration_path
    source=official_source(args.official) if args.official else None
    with Run(args.output,dict(inputs={k:sha(Path(v)) for k,v in paths.items()},model=args.model,
                             registry=registry,official=source),args.resume) as run:
        state=run.state
        def prepare_inputs():
            pairs=read_jsonl(input_path)
            official, extra=None, []
            if args.official:
                official, metadata=load_official(pairs,args.official)
                atomic(state/'official_metadata.json',metadata)
                jsonl(state/'official_matched.jsonl',official)
                extra=[state/'official_metadata.json',state/'official_matched.jsonl']
                print(f"Official {args.official}: matched {len(official)} / {metadata['available_questions']} questions",flush=True)
            development=read_jsonl(calibration_path) if calibration_path is not None else None
            return prepare(pairs,development,official,registry,args.model,state)+extra
        run.stage('prepare',prepare_inputs)
        for mode in (('calibration','full') if calibration_path is not None else ('full',)):
            def evaluate(mode=mode):
                destination=state/mode
                command=[sys.executable,'-m','common.model_eval.run','--protocol',str(state/'protocol.json'),
                    '--requests',str(state/(mode+'.jsonl')),'--model-key',args.model,'--mode',mode,
                    '--output-dir',str(destination/'generation')]
                if mode=='full' and calibration_path is not None:
                    command+=['--calibration-gate',str(state/'calibration/analysis/calibration_gate.json')]
                subprocess.run(command,env=child_environment(),check=True)
                return [destination/'generation/outputs.jsonl',destination/'generation/run_manifest.json']
            run.stage(mode+'_generation',evaluate)
            def analyze(mode=mode):
                destination=state/mode
                subprocess.run([sys.executable,'-m','common.model_eval.analyze','--protocol',str(state/'protocol.json'),
                    '--requests',str(state/(mode+'.jsonl')),'--model-key',args.model,'--mode',mode,
                    '--outputs',str(destination/'generation/outputs.jsonl'),
                    '--run-manifest',str(destination/'generation/run_manifest.json'),
                    '--output-dir',str(destination/'analysis')],env=child_environment(),check=True)
                if mode=='calibration' and not read(destination/'analysis/calibration_gate.json')['full_run_allowed']:
                    raise ValueError('Model did not pass the response-format calibration')
                return list((destination/'analysis').glob('*json*'))
            run.stage(mode+'_analysis',analyze)
        def publish():
            report=read(state/'full/analysis/analysis.json')
            summary=dict(model=registry['models'][args.model]['model_id'],
                         results=report['counts'],tokens=report['generated_token_count'])
            if args.official:
                meta=read(state/'official_metadata.json')
                summary['official']={key:meta[key] for key in
                    ('repo_id','family','available_questions','matched_questions','scope')}
            atomic(run.output/'summary.json',summary)
            with (run.output/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
                table=result_table(summary)
                writer=csv.DictWriter(stream,fieldnames=list(table[0]))
                writer.writeheader();writer.writerows(table)
            (run.output/'summary.txt').write_text(format_results(summary),encoding='utf-8')
            rows=read_jsonl(state/'full/analysis/normalized_outputs.jsonl')
            keys=['question_id','view','gold_answer','intended_target','generated_text','pred_answer','scientific_role',
                  'answer_correct','answer_valid','generated_token_count','finish_reason','max_token_cutoff']
            jsonl(run.output/'predictions.jsonl',[{k:row[k] for k in keys if k in row} for row in rows])
            return [run.output/name for name in ('summary.json','summary.csv','summary.txt','predictions.jsonl')]
        run.stage('results',publish)
        print((run.output/'summary.txt').read_text(encoding='utf-8'))
        print('Saved: '+str(run.output))


def main():
    registry=read(MODEL_CONFIG)
    parser=argparse.ArgumentParser(description='Evaluate MedQA, official MedDistractQA, and Hard data and summarize results')
    parser.add_argument('--input',required=True,help='Hard dataset JSONL or generation output directory')
    parser.add_argument('--calibration',help=argparse.SUPPRESS)
    parser.add_argument('--official',choices=tuple(OFFICIAL_FILES),
                        help='Download the official HF dataset and match it to the input Hard questions')
    parser.add_argument('--model',choices=registry['primary_model_keys'],default='llama31_8b_instruct',
                        help='Model configuration key; llama31_8b_instruct selects meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())


if __name__ == "__main__":
    main()
