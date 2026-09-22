"""Convert frozen E08/E11 input payloads into a standalone input manifest."""
import argparse
from robust_medical_qa.io import read_json, write_json, require, digest


def convert(e08, e11):
    base = read_json(e08)
    control = read_json(e11)
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
    return {'generation': base['cfg'], 'rows': rows, 'source_sha256': {'e08': digest(e08), 'e11': digest(e11)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e08', required=True)
    parser.add_argument('--e11', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    write_json(args.output, convert(args.e08, args.e11))


if __name__ == '__main__':
    main()
