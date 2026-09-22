"""Recompute attention counts from the shared, text-free outcome records."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_tables():
    expected = json.loads((ROOT / 'results/attention_summary.json').read_text())
    rows = list(csv.DictReader((ROOT / 'results/attention_outcomes.csv').open()))
    assert len(rows) == expected['rows'] == 480
    records = {(r['question_id'], r['condition']): r for r in rows}
    assert len(records) == len(rows), 'Duplicate question/condition'
    base = {r['question_id']: r for r in rows if r['condition'] == 'baseline'}
    assert len(base) == expected['n_independent_questions'] == 96
    assert {r['condition'] for r in rows} == set(expected['conditions'])
    result = {}
    for condition, frozen in expected['conditions'].items():
        group = [r for r in rows if r['condition'] == condition]
        assert {r['question_id'] for r in group} == set(base)
        result[condition] = dict(
            n=len(group), correct=sum(r['role'] == 'G' for r in group),
            target=sum(r['role'] == 'T' for r in group),
            invalid=sum(r['answer_valid'] == '0' for r in group),
            gain=sum(r['role'] == 'G' and base[r['question_id']]['role'] != 'G' for r in group),
            loss=sum(r['role'] != 'G' and base[r['question_id']]['role'] == 'G' for r in group))
        assert result[condition] == frozen, (condition, result[condition], frozen)
    probe = json.loads((ROOT / 'results/role_probe_exact.json').read_text())
    for row in probe['rows']:
        assert sum(row['correct_per_seed']) == row['total_correct']
        assert row['denominator_per_seed'] == 2 * probe['independent_items']
        assert row['repeated_predictions'] == len(probe['seeds']) * row['denominator_per_seed']
        assert abs(row['total_correct'] / row['repeated_predictions'] - row['mean_accuracy']) < 1e-12
    return result


if __name__ == '__main__':
    results = check_tables()
    print('| Condition | Correct / 96 | Wrong→correct | Correct→wrong |')
    print('|---|---:|---:|---:|')
    for condition in ('baseline', 'distractor_option_cut', 'distractor_all_source_cut',
                      'clinical_option_cut', 'clinical_all_source_cut'):
        r = results[condition]
        print(f"| {condition} | {r['correct']} | {r['gain']} | {r['loss']} |")
    print('\nPASS: 480 outcomes, 96 paired questions, and probe count arithmetic.')
    print('Transmission values are quoted from rounded original output, not recalculated here.')
