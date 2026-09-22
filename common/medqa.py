"""Pinned MedQA loading and content-based alignment with existing Hard inputs."""
import re
from pathlib import Path

from common.checkpoints import read, sha
from common.io import read_jsonl, require

REPO = 'GBaker/MedQA-USMLE-4-options-hf'
REVISION = '17af9355ef89fba60de966eabaeba797c695f86e'
FILES = {split: split + '.json' for split in ('train', 'dev', 'test')}
TEST_ORDER = Path(__file__).resolve().parent / 'configs/medqa_test_order.json'


def medqa_source(split):
    require(split in FILES, 'MedQA split must be train, dev, or test')
    return dict(repo_id=REPO, revision=REVISION, filename=FILES[split], split=split)


def content_key(question, options, answer):
    # Exact text and option order preserve the original inference prompts.
    return question, tuple(options[c] for c in 'ABCD'), answer


def normalize_rows(rows, split):
    require(isinstance(rows, list) and bool(rows), 'MedQA data must be nonempty')
    medqa_source(split)
    required = {'id', 'sent1', 'sent2', 'label'} | {f'ending{i}' for i in range(4)}
    result, ids, content = [], set(), set()
    for position, row in enumerate(rows):
        require(isinstance(row, dict) and required <= row.keys(), f'Invalid MedQA fields at row {position}')
        sid = row['id']
        match = re.fullmatch(split + r'-(\d+)', sid) if isinstance(sid, str) else None
        require(match is not None and sid not in ids, f'Invalid or duplicate MedQA ID at row {position}')
        require(isinstance(row['sent1'], str) and row['sent1'].strip(), f'Empty MedQA question: {sid}')
        require(row['sent2'] == '', f'Unexpected additional MedQA question text: {sid}')
        require(type(row['label']) is int and 0 <= row['label'] < 4, f'Invalid MedQA label: {sid}')
        options = {c: row[f'ending{i}'] for i, c in enumerate('ABCD')}
        require(all(isinstance(v, str) and v.strip() for v in options.values()), f'Invalid MedQA options: {sid}')
        answer = 'ABCD'[row['label']]
        key = content_key(row['sent1'], options, answer)
        require(key not in content, f'Duplicate MedQA question/options/answer: {sid}')
        ids.add(sid)
        content.add(key)
        result.append(dict(idx=int(match.group(1)), source_id=sid, question=row['sent1'],
                           options=options, answer_idx=answer))
    return result


def construction_order(rows, order):
    """Restore the existing test source IDs and iteration order without changing text."""
    by_id = {r['source_id']: r for r in rows}
    require(len(order) == len(rows) and {r['hf_id'] for r in order} == set(by_id)
            and {r['idx'] for r in order} == set(range(len(rows))), 'Invalid MedQA test source mapping')
    return [dict(by_id[r['hf_id']], idx=r['idx'], source_id=f"test-{r['idx']:05d}") for r in order]


def load_medqa(split='test', *, construction=False):
    from huggingface_hub import hf_hub_download
    source = medqa_source(split)
    path = Path(hf_hub_download(repo_id=source['repo_id'], repo_type='dataset',
        filename=source['filename'], revision=source['revision'], token=False))
    rows = normalize_rows(read_jsonl(path), split)
    metadata = dict(source, file_sha256=sha(path), available_questions=len(rows))
    if construction and split == 'test':
        order = read(TEST_ORDER)
        rows = construction_order(rows, order)
        metadata['alignment'] = [dict(source_id=r['source_id'], hf_id=o['hf_id']) for r, o in zip(rows, order)]
    return rows, metadata


def match_medqa(rows, pairs):
    """Use HF Clean text while retaining Hard question IDs, source indices, and order."""
    from common.model_eval.core import validate_source_pair
    require(bool(pairs), 'Empty evaluation pairs')
    index = {}
    for row in rows:
        key = content_key(row['question'], row['options'], row['answer_idx'])
        require(key not in index, 'Ambiguous MedQA question/options/answer')
        index[key] = row
    matched, mapping, ids, used = [], [], set(), set()
    for pair in pairs:
        validate_source_pair(pair)
        sid = pair['question_id']
        require(sid not in ids, f'Duplicate Hard question ID: {sid}')
        ids.add(sid)
        key = content_key(pair['clean_question'], pair['options'], pair['gold_answer'])
        source = index.get(key)
        require(source is not None, f'No exact MedQA question/options/answer match for {sid}; stopped before inference')
        require(source['source_id'] not in used, f'Duplicate Hard source question: {sid}')
        used.add(source['source_id'])
        matched.append(dict(pair, clean_question=source['question'], options=source['options'],
                            gold_answer=source['answer_idx']))
        mapping.append(dict(question_id=sid, hf_id=source['source_id']))
    return matched, mapping
