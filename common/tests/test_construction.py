"""Boundary cases shared by the consolidated construction stages."""
import unittest

from common.hard_data.prompting import HARMONY_FINAL_ANSWER_PREFIX
from common.hard_data.runtime import sha256_text
from common.hard_data.judge import target_gate_result
from common.hard_data.scoring import as_token_ids, build_final_answer_token_plan, render_user_turn_tokens
from common.hard_data.selection import decide_candidate_with_residual_policy


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def decode(self, tokens, **kwargs):
        return ''.join(chr(token) for token in tokens)

    def apply_chat_template(self, messages, **kwargs):
        if 'reasoning_effort' in kwargs:
            raise TypeError('This tokenizer expects nested template kwargs')
        return {'input_ids': [self.encode('user message')]}


class ConstructionTests(unittest.TestCase):
    def test_harmony_tokens_and_template_fallback(self):
        tokenizer = CharacterTokenizer()
        plan = build_final_answer_token_plan(tokenizer)
        self.assertEqual(plan['decoded_common_prefix'], HARMONY_FINAL_ANSWER_PREFIX)
        self.assertEqual(plan['candidate_token_ids'], {c: ord(c) for c in 'ABCD'})
        self.assertEqual(render_user_turn_tokens(tokenizer, []), tokenizer.encode('user message'))
        self.assertEqual(as_token_ids({'input_ids': [(1, 2)]}), [1, 2])
        with self.assertRaises(ValueError):
            as_token_ids([[1], [2]])

    def test_reject_non_atomic_answer_tokens(self):
        class SplitAnswerTokenizer(CharacterTokenizer):
            def encode(self, text, **kwargs):
                tokens = super().encode(text, **kwargs)
                return tokens + [0] if text.endswith('D') else tokens
        with self.assertRaisesRegex(RuntimeError, 'single-token'):
            build_final_answer_token_plan(SplitAnswerTokenizer())

    def test_gate_rejects_gold_in_either_order(self):
        good = {'strength': 'strong', 'primary_evoked_option': 'B',
                'evoked_options': ['B'], 'relation_type': 'SEMANTIC'}
        bad = {**good, 'evoked_options': ['A', 'B'], 'relation_type': 'MULTI'}
        self.assertTrue(target_gate_result({'forward': {'parsed': good}, 'reverse': {'parsed': good}}, 'B', 'A')['pass'])
        for forward, reverse in [(good, bad), (bad, good), (good, None)]:
            self.assertFalse(target_gate_result({'forward': {'parsed': forward}, 'reverse': {'parsed': reverse}}, 'B', 'A')['pass'])

    def test_duplicate_ties_and_infeasible_sources(self):
        base = {'candidate_text_sha256': 'same-text', 'intended_target': 'B',
                'target_gate_label': 'INTENDED_SINGLE_TARGET',
                'option_logprobs': {'A': -2, 'B': -1, 'C': -4, 'D': -4}}
        candidates = [{**base, 'candidate_id': 'later', 'candidate_rank': 3},
                      {**base, 'candidate_id': 'earlier', 'candidate_rank': 1}]
        for rows in [candidates, candidates[::-1]]:
            result, _ = decide_candidate_with_residual_policy(rows, 'A', 'A', 'source')
            self.assertEqual(result.selected_candidate_id, 'earlier')
            self.assertEqual(result.resolution_method, 'duplicate_text_canonical_provenance')
            self.assertTrue(result.selected_by_flip_priority)
        result, _ = decide_candidate_with_residual_policy([], 'A', 'A', 'source')
        self.assertEqual(result.final_status, 'excluded_construction_infeasible')



if __name__ == '__main__':
    unittest.main()
