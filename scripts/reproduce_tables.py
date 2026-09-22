"""Recompute attention counts from the shared, text-free outcome records."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robust_medical_qa.evaluation import outcome_counts


def check_tables():
    expected = json.loads((ROOT / 'results/attention_summary.json').read_text())
    rows = list(csv.DictReader((ROOT / 'results/attention_outcomes.csv').open()))
    assert len(rows) == expected['rows']
    result = outcome_counts(rows)
    assert result == expected['conditions'], 'Outcome counts differ'
    probe = json.loads((ROOT / 'results/role_probe_exact.json').read_text())
    for row in probe['rows']:
        assert sum(row['correct_per_seed']) == row['total_correct']
        assert row['denominator_per_seed'] == 2 * probe['independent_items']
        assert row['repeated_predictions'] == len(probe['seeds']) * row['denominator_per_seed']
        assert abs(row['total_correct'] / row['repeated_predictions'] - row['mean_accuracy']) < 1e-12
    return result


if __name__ == '__main__':
    results = check_tables()
    n = results['baseline']['n']
    print(f'| Condition | Correct / {n} | Wrong→correct | Correct→wrong |')
    print('|---|---:|---:|---:|')
    for condition in ('baseline', 'distractor_option_cut', 'distractor_all_source_cut',
                      'clinical_option_cut', 'clinical_all_source_cut'):
        r = results[condition]
        print(f"| {condition} | {r['correct']} | {r['gain']} | {r['loss']} |")
    print(f'\nPASS: {sum(r["n"] for r in results.values())} outcomes, {n} paired questions, and probe count arithmetic.')
    print('Transmission values are quoted from rounded original output, not recalculated here.')
