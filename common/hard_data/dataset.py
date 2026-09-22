"""Build paired datasets from the fixed accepted candidate pool."""
import math
from common.checkpoints import sample_by_hash
from .scoring import DEFAULT_REVISION
from .runtime import insert_before_final_sentence
from .selection import decide_candidate_with_residual_policy

def validate_scores(requests, scores):
    ids = [r['request_id'] for r in scores]
    if len(set(ids)) != len(ids) or set(ids) != {r['request_id'] for r in requests} or len(scores) != len(requests):
        raise ValueError('Score coverage or duplicate mismatch')
    for row in scores:
        values = row.get('option_logprobs', {})
        if (row.get('status') != 'ok' or row.get('model_revision') != DEFAULT_REVISION
                or set(values) != set('ABCD') or not all(math.isfinite(v) for v in values.values())):
            raise ValueError('Invalid model score')
    return {r['request_id']: r for r in scores}

def select_dataset(sources, pool, scores, namespace="nl-dev890-v1", random_salt="nl-random-same-target-v1"):
    index = validate_scores(pool['score_requests'], scores)
    grouped = {s['source_id']: [] for s in sources}
    for can in pool['valid_candidates']:
        grouped[can['source_id']].append(dict(can, option_logprobs=index[can['candidate_id']]['option_logprobs']))
    output, audits = [], []
    for source in sources:
        sid, gold = source['source_id'], source['answer_idx']
        candidates = grouped[sid]
        decision, _ = decide_candidate_with_residual_policy(candidates, index['clean:'+sid]['top1'], gold, sid)
        audits.append(dict(source_id=sid, **decision.__dict__))
        if decision.selected_candidate_id is None:
            continue
        by_id = {r['candidate_id']:r for r in candidates}
        hard = by_id[decision.selected_candidate_id]
        chosen = dict(hard=hard, random_valid=by_id[pool['random_selection'][sid]],
            random_same_target=sample_by_hash([r for r in candidates if r['intended_target']==hard['intended_target']],
                                                sid, random_salt))
        for view in ('clean','random_valid','hard','random_same_target'):
            can = chosen.get(view)
            output.append(dict(request_id=f'{namespace}:{sid}:{view}', source_id=sid, source_idx=source['idx'],
                view=view, question=insert_before_final_sentence(source['question'],can['sentence']) if can else source['question'],
                clean_question=source['question'], options=source['options'], gold_answer=gold,
                intended_target=can['intended_target'] if can else hard['intended_target'],
                candidate_id=can['candidate_id'] if can else None,
                target_gate_label=can['target_gate_label'] if can else None,
                target_set=sorted(set(sum([can['judgments'][tag]['parsed']['evoked_options'] for tag in ('forward','reverse')],[]))) if can else [],
                added_distractor=can['sentence'] if can else '', must_not_be_used_for_training=True))
    return output, audits


def paired_rows(sources, views, audits):
    audits = {s['source_id']: s for s in audits}
    output = []
    for row in views:
        if row['view'] != 'hard': continue
        audit = audits[row['source_id']]
        output.append(dict(question_id=row['source_id'], source_idx=row['source_idx'],
            clean_question=row['clean_question'], distracted_question=row['question'],
            options=row['options'], gold_answer=row['gold_answer'], intended_target=row['intended_target'],
            added_distractor=row['added_distractor'], candidate_id=row['candidate_id'],
            selection_pool='harmful_flip_priority' if audit['selected_by_flip_priority'] else 'maximum_D_fallback',
            construction_harmful_flip=audit['selected_by_flip_priority'], must_not_be_used_for_training=True))
    return output
