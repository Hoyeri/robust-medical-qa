"""Regression checks for preserved probe semantics and portable data handling."""
import ast
import csv
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from common.io import read_json, read_jsonl, hard_dataset_path, hard_dataset_filename
from common.checkpoints import atomic
from common.probes import ProbeConfig, cross_validate, load_paired
from common.evaluation import outcome_counts, paired_summary
from common.spans import phrase_span_in, finding_source, attribution_metadata

ROOT = Path(__file__).resolve().parents[2]


def original_functions(path):
    tree = ast.parse((ROOT / path).read_text())
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    namespace = {'np': np, 'K': 4}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


class CoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(818)
        self.a = rng.normal(size=(19, 9)).astype(np.float32)
        self.b = rng.normal(size=(19, 9)).astype(np.float32)
        self.b[:, :2] += .7

    def test_original_probe_and_label_shuffle(self):
        old = original_functions('common/tests/reference/probe_contextual_role.py')
        cfg = ProbeConfig(components=4)
        for shuffle in (False, True):
            expected = old['cv_accuracy'](self.a, self.b, shuffle_labels=shuffle)[0]
            observed = cross_validate(self.a, self.b, cfg, shuffle='labels' if shuffle else None)
            self.assertAlmostEqual(observed['mean_accuracy'], expected, places=14)
            self.assertTrue(all(r['n'] == 2 * len(self.a) for r in observed['repeats']))

    def test_historical_grouped_profiles(self):
        groups = ['a'] * 6 + ['b'] * 4 + ['c'] * 4 + ['d'] * 3 + ['e'] * 2
        for name, aggregation, sanitize in [('role/probe_robustness.py','pooled',False),
                                             ('transmission/analyze_transmission.py','fold_mean',True)]:
            old = original_functions({'role/': 'common/tests/reference/', 'transmission/': 'common/tests/reference/'}[name.split('/')[0]+'/'] + name.split('/',1)[1])
            cfg = ProbeConfig(components=4, standard_clip=20, logit_clip=30, nan_to_num=sanitize, aggregation=aggregation)
            expected = np.mean([old['grouped_cv'](self.a, self.b, groups, seed=s) for s in cfg.seeds])
            actual = cross_validate(self.a, self.b, cfg, groups)
            self.assertAlmostEqual(actual['mean_accuracy'], expected, places=14)
            if sanitize:
                expected_shuffle = np.mean([old['grouped_cv'](self.a, self.b, groups, seed=s, shuffle=True) for s in cfg.seeds])
                self.assertAlmostEqual(cross_validate(self.a,self.b,cfg,groups,shuffle='pair_swap')['mean_accuracy'], expected_shuffle, places=14)

    def test_frozen_attention_outcomes(self):
        with (ROOT / 'common/tests/fixtures/attention_outcomes.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(outcome_counts(rows), read_json(ROOT / 'common/tests/fixtures/attention_summary.json')['conditions'])
        with self.assertRaises(ValueError):
            outcome_counts(rows + [rows[0]])

    def test_unknown_template_requires_explicit_span(self):
        with self.assertRaisesRegex(ValueError, 'Unknown finding template'):
            phrase_span_in('A new relation observed signal red.')
        sentence = 'A new relation observed signal red.'
        item = {'source_annotations': {'H1': {'sentence': sentence, 'phrase_char_span': [24,34]}}}
        self.assertEqual(finding_source(item,'H1'),(sentence,(24,34)))
        known = 'Her coworker also has signal red.'
        a,b = phrase_span_in(known)
        self.assertEqual(known[a:b], 'signal red')

    def test_unknown_family_is_not_silently_classified(self):
        rows = [{'item_id':'i', 'meta':{'cue':{'hpo':'c'}}, 'variants':{'C':'Base.', 'H1':'Base. Her sibling has signal red.'}}]
        with self.assertRaisesRegex(ValueError, 'Unknown attribution family'):
            attribution_metadata(['i'],rows)

    def test_imports_have_no_execution_side_effects(self):
        import importlib
        for name in ['experiments.build_dataset','experiments.evaluate_models','experiments.attribution_probe',
                     'experiments.transmission','experiments.head_search','experiments.attention_blocking']:
            self.assertTrue(importlib.import_module(name))

    def test_pair_loader_rejects_extra_or_duplicate_variants(self):
        with tempfile.TemporaryDirectory() as d:
            for variant in ('Ma', 'H1', 'Ua'):
                np.savez(Path(d)/f'{variant}.npz', item_ids=['a', 'b'], h=np.ones((2, 2, 4)))
            for variants in ([], ['Ma'], ['Ma', 'H1', 'Ua'], ['Ma', 'Ma']):
                with self.subTest(variants=variants), self.assertRaisesRegex(ValueError, 'two distinct variants'):
                    load_paired(d, variants, ['h'])

    def test_dataset_directory_and_legacy_file_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaisesRegex(ValueError, 'Expected one Hard dataset'):
                hard_dataset_path(root)
            legacy = root/'pairs.jsonl'
            legacy.write_text('{}\n')
            self.assertEqual(hard_dataset_path(root), legacy)
            bystander = root/hard_dataset_filename('bystander')
            bystander.write_text('{}\n')
            self.assertEqual(hard_dataset_path(root), bystander)
            self.assertEqual(hard_dataset_path(legacy), legacy)
            (root/hard_dataset_filename('nonliteral')).write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'Expected one Hard dataset'):
                hard_dataset_path(root)
            self.assertEqual(hard_dataset_path(bystander), bystander)

    def test_included_attribution_data_has_original_comparisons(self):
        items = read_jsonl(ROOT/'common/attribution_data.jsonl')
        self.assertEqual(len(items),118)
        paired = [it for it in items if 'Ma' in it['variants'] and 'H1' in it['variants']]
        self.assertEqual(len(paired),53)
        meta = attribution_metadata([it['item_id'] for it in paired],items)
        self.assertEqual(len(set(meta['concept'])),35)
        self.assertEqual(meta['family'].count('coworker'),39)
        self.assertEqual(meta['family'].count('classmate_child'),14)
        for item in paired:
            findings=[]
            for variant in ('Ma','H1'):
                sentence,(a,b)=finding_source(item,variant)
                findings.append(sentence[a:b])
            self.assertEqual(*findings)

    def test_pair_loader_preserves_intersection_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            a = np.arange(24).reshape(3, 2, 4)
            b = np.arange(16).reshape(2, 2, 4)
            np.savez(Path(d)/'Ma.npz', item_ids=['a', 'b', 'c'], h=a)
            np.savez(Path(d)/'H1.npz', item_ids=['c', 'a'], h=b)
            ids, A, B = load_paired(d, ['Ma', 'H1'], ['h'])
            self.assertEqual(ids, ['a', 'c'])
            np.testing.assert_array_equal(A['h'], a[[0, 2]])
            np.testing.assert_array_equal(B['h'], b[[1, 0]])

    def test_bootstrap_rejects_nonfinite_observations(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Non-finite'):
                paired_summary([0.2, value], repeats=40, seed=1)

    def test_transmission_indices_and_paired_denominators(self):
        from experiments.transmission import analyze
        rng = np.random.default_rng(51)
        ids = [f'i{i}' for i in range(6)]
        items = [dict(item_id=i, meta={'cue': {'hpo': i}, 'attribution_family': 'coworker' if n < 3 else 'classmate_child'})
                 for n, i in enumerate(ids)]
        # Additional input items need not have both Ma and H1 representations.
        items.append(dict(item_id='unpaired'))
        config = read_json(ROOT/'common/configs/transmission.json')['analysis']
        config.update(indices={'h': [0, 1]}, summary_indices=[0, 1], sensitivity_layers=[0, 1],
                      bootstrap={'repeats': 40, 'seed': 1}, contrasts=[['Ma', 'H1'], ['Ma', 'Ua']])
        config['probe'].update(components=2, folds=2, seeds=[0])
        source = dict(A_all=[0.1, 0.2], A_ans=[0.05, 0.1])
        record = dict(q=0.3, sources={s: copy.deepcopy(source) for s in ('sent', 'phr', 'rand')})
        alignment = {i: {v: copy.deepcopy(record) for v in ('Ma', 'H1')} for i in ids}
        for i in ids[:2]:
            alignment[i]['Ua'] = copy.deepcopy(record)
        alignment['unpaired'] = {'Ua': copy.deepcopy(record)}
        with tempfile.TemporaryDirectory() as d:
            for variant in ('Ma', 'H1'):
                np.savez(Path(d)/f'{variant}.npz', item_ids=ids, h=rng.normal(size=(6, 2, 4)))
            atomic(Path(d)/'alignment.json', alignment)
            result = analyze(d, items, config)
            self.assertEqual(result['independent_pairs'], 6)
            self.assertEqual(result['paired_answer_score']['mean']['n'], 6)
            self.assertTrue(all(r['n'] == (2 if r['b'] == 'Ua' else 6) for r in result['sensitivity']))
            for key, value in [('indices', {'h': [-1]}), ('indices', {'h': [2]}),
                               ('summary_indices', []), ('summary_indices', [0, 0]),
                               ('sensitivity_layers', []), ('sensitivity_layers', [-1]),
                               ('sensitivity_layers', [0, 0]), ('sensitivity_layers', [2])]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    analyze(d, items, dict(config, **{key: value}))
            del alignment['i0']['H1']['q']
            atomic(Path(d)/'alignment.json', alignment)
            with self.assertRaisesRegex(ValueError, 'Missing paired answer score: i0'):
                analyze(d, items, config)


if __name__ == '__main__':
    unittest.main()
