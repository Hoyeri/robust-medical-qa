"""Regression checks for preserved probe semantics and portable data handling."""
import ast
import csv
from dataclasses import replace
from pathlib import Path
import unittest
import numpy as np
from robust_medical_qa.io import read_json
from robust_medical_qa.probes import ProbeConfig, cross_validate
from robust_medical_qa.evaluation import outcome_counts
from robust_medical_qa.spans import phrase_span_in, finding_source, attribution_metadata

ROOT = Path(__file__).resolve().parents[1]


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
        old = original_functions('reference/role/probe_contextual_role.py')
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
            old = original_functions('reference/' + name)
            cfg = ProbeConfig(components=4, standard_clip=20, logit_clip=30, nan_to_num=sanitize, aggregation=aggregation)
            expected = np.mean([old['grouped_cv'](self.a, self.b, groups, seed=s) for s in cfg.seeds])
            actual = cross_validate(self.a, self.b, cfg, groups)
            self.assertAlmostEqual(actual['mean_accuracy'], expected, places=14)
            if sanitize:
                expected_shuffle = np.mean([old['grouped_cv'](self.a, self.b, groups, seed=s, shuffle=True) for s in cfg.seeds])
                self.assertAlmostEqual(cross_validate(self.a,self.b,cfg,groups,shuffle='pair_swap')['mean_accuracy'], expected_shuffle, places=14)

    def test_frozen_attention_outcomes(self):
        with (ROOT / 'results/attention_outcomes.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(outcome_counts(rows), read_json(ROOT / 'results/attention_summary.json')['conditions'])
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
        for name in ['attribution_probe','transmission_audit','extract_representations','attention_blocking','prepare_attention']:
            self.assertTrue(importlib.import_module('experiments.'+name))


if __name__ == '__main__':
    unittest.main()
