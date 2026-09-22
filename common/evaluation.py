"""Outcome aggregation and paired bootstrap summaries."""
import random
import statistics
import math
from .io import require


def outcome_counts(rows, baseline='baseline'):
    records = {(r['question_id'], r['condition']): r for r in rows}
    require(len(records) == len(rows), 'Duplicate question/condition')
    base = {r['question_id']: r for r in rows if r['condition'] == baseline}
    require(bool(base), 'Missing baseline')
    result = {}
    for condition in dict.fromkeys(r['condition'] for r in rows):
        group = [r for r in rows if r['condition'] == condition]
        require({r['question_id'] for r in group} == set(base), 'Condition inventory mismatch')
        result[condition] = dict(n=len(group), correct=sum(r['role'] == 'G' for r in group),
            target=sum(r['role'] == 'T' for r in group),
            invalid=sum(str(r['answer_valid']) in ('0', 'False') for r in group),
            gain=sum(r['role'] == 'G' and base[r['question_id']]['role'] != 'G' for r in group),
            loss=sum(r['role'] != 'G' and base[r['question_id']]['role'] == 'G' for r in group))
    return result


def paired_summary(values, repeats, seed, statistic='median'):
    require(len(values) > 0 and repeats >= 40, 'Insufficient bootstrap input/repeats')
    require(all(math.isfinite(v) for v in values), 'Non-finite paired observations')
    estimator = {'median': statistics.median, 'mean': statistics.mean}[statistic]
    rng = random.Random(seed)
    samples = sorted(estimator(rng.choices(values, k=len(values))) for _ in range(repeats))
    return {'n': len(values), 'estimate': estimator(values), 'ci95': [samples[int(.025 * repeats)], samples[int(.975 * repeats)]]}
