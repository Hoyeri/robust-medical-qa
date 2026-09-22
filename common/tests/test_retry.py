import copy
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common.checkpoints import sample_by_hash, sha
from common.hard_data.judge import target_gate_result as gate, JUDGE_PROMPT, parse_judgment
from common.hard_data.retry import RetryPolicy, round_seed, run_target, classify_failure, build_scoring_pool
from common.hard_data.runtime import insert_before_final_sentence
from common.hard_data.selection import decide_candidate_with_residual_policy

SOURCE = dict(source_id='unit-1', idx=1, question='A person arrived. Which answer is best?',
              options=dict(A='Gold', B='Target', C='Other', D='Other two'), answer_idx='A')
GEN = '{question}\nTopic: {clinical_topic}'
JUDGE = '{question}\n{choices}\n{sentence}'


def judgment(kind='accepted'):
    p=dict(primary_evoked_option='B', evoked_options=['B'], relation_type='SEMANTIC', strength='strong')
    if kind == 'cue': p.update(primary_evoked_option='C', evoked_options=['C'])
    if kind == 'none': p.update(primary_evoked_option='NONE', evoked_options=[], strength='none')
    if kind == 'gold': p.update(primary_evoked_option='MULTI', evoked_options=['A','B'], relation_type='MULTI')
    if kind == 'multi': p.update(primary_evoked_option='MULTI', evoked_options=['B','C'], relation_type='MULTI')
    if kind == 'relation_none': p.update(relation_type='NONE')
    # Extra legacy labels must not affect the pure cue decision, even if present
    # as audit metadata. The real judge is no longer asked to produce them.
    if kind == 'content': p.update(usage_type='MIXED')
    if kind == 'clinical': p.update(patient_clinical_relevance='DIRECT', primary_evoked_option='C', evoked_options=['C'])
    return dict(parsed=None if kind=='parse' else p, error='bad json' if kind=='parse' else None,
                finish_reason='length' if kind=='length' else 'stop', raw='fixture', tokens=[])


class Backend:
    def __init__(self, batches, fail_at=None):
        self.batches=copy.deepcopy(batches); self.gcalls=[]; self.jcalls=0; self.records={}; self.fail_at=fail_at
    def generate(self, prompt, n, seed):
        batch=self.batches[len(self.gcalls)]
        if len(batch)!=n: raise AssertionError((len(batch),n))
        self.gcalls.append((n,seed)); out=[]
        for kind in batch:
            i=len(self.records); self.records[i]=kind
            out.append(dict(sentence=f'The patient joked that the printer was example c{i}.',
                            raw_generation='fixture', generated_token_ids=[1,2],
                            finish_reason='length' if kind=='genlength' else 'stop'))
        return out
    def judge(self, prompt):
        self.jcalls+=1
        if self.jcalls==self.fail_at: raise RuntimeError('simulated interruption')
        i=int(re.search(r'example c(\d+)',prompt)[1]); kind=self.records[i]
        if isinstance(kind,tuple): kind=kind[0 if prompt.splitlines()[1].startswith('A.') else 1]
        return judgment(kind)


class RetryTests(unittest.TestCase):
    def run_case(self,batches,policy=None):
        b=Backend(batches); p=policy or RetryPolicy(slots=len(batches[0]))
        with tempfile.TemporaryDirectory() as d:
            result=run_target(SOURCE,'B',GEN,JUDGE,p,d,b.generate,b.judge,gate,{'revision':'fixed'})
        return result,b
    def test_all_pass_no_extra_calls(self):
        r,b=self.run_case([['accepted']*8]); self.assertEqual(len(b.gcalls),1); self.assertEqual(r['accepted_n'],8)
    def test_cue_only_holes(self):
        r,b=self.run_case([['accepted','cue','content','parse'],['cue'],['accepted']])
        self.assertEqual([n for n,_ in b.gcalls],[4,1,1]); self.assertEqual(r['accepted_n'],3)
        self.assertEqual([s['status'] for s in r['slots']],['accepted','accepted','accepted','technical'])
    def test_worst_case_cap(self):
        r,b=self.run_case([['cue']*8]*3); self.assertEqual(r['generated_n'],24)
        self.assertEqual(r['accepted_n'],0); self.assertTrue(all(s['status']=='option_cue_budget_exhausted' for s in r['slots']))
    def test_clinical_metadata_does_not_block_cue_retry(self):
        r,b=self.run_case([[('cue','clinical')],['accepted']]); self.assertEqual(len(b.gcalls),2); self.assertEqual(r['slots'][0]['status'],'accepted')
    def test_technical_supersedes_content(self):
        r,b=self.run_case([[('content','parse')]]); self.assertEqual(r['slots'][0]['status'],'technical')
    def test_generation_truncation_no_judge(self):
        r,b=self.run_case([['genlength']]); self.assertEqual(b.jcalls,0); self.assertEqual(r['slots'][0]['status'],'technical')
    def test_judge_truncation_no_retry(self):
        r,b=self.run_case([['length']]); self.assertEqual(len(b.gcalls),1); self.assertEqual(r['slots'][0]['status'],'technical')
    def test_gold_free_multi_still_passes(self):
        r,b=self.run_case([[('accepted','multi')]]); self.assertEqual(r['valid_candidates'][0]['target_gate_label'],'GOLD_FREE_MULTI')
    def test_gold_cue_is_retryable(self):
        r,b=self.run_case([['gold'],['accepted']]); self.assertEqual([n for n,_ in b.gcalls],[1,1])
    def test_no_cue_is_retryable(self):
        r,b=self.run_case([['none'],['accepted']]); self.assertEqual(r['accepted_n'],1)
    def test_usage_metadata_does_not_reject_accepted_cue(self):
        r,b=self.run_case([['cue'],['content']]); self.assertEqual(len(b.gcalls),2); self.assertEqual(r['shortage_n'],0)
    def test_no_extra_rounds(self):
        r,b=self.run_case([['cue']],RetryPolicy(slots=1,extra_rounds=0)); self.assertEqual(r['generated_n'],1)
    def test_seed_deterministic_and_distinct(self):
        p=RetryPolicy(); seeds=[round_seed(p,'s','B',r) for r in range(3)]
        self.assertEqual(seeds[0],42); self.assertEqual(len(set(seeds)),3)
        self.assertEqual(seeds,[round_seed(p,'s','B',r) for r in range(3)])
        self.assertNotEqual(seeds[1],round_seed(p,'s','C',1))
    def test_unique_ids_and_same_slot(self):
        r,b=self.run_case([['cue','cue'],['cue','accepted'],['accepted']])
        ids=[a['candidate']['candidate_id'] for a in r['attempts']]
        self.assertEqual(len(ids),len(set(ids))); self.assertEqual([a['candidate']['slot'] for a in r['attempts']],[0,1,0,1,0])
    def test_resume_no_repeated_saved_inference(self):
        p=RetryPolicy(slots=2); b=Backend([['accepted','accepted']],fail_at=3)
        with tempfile.TemporaryDirectory() as d:
            args=(SOURCE,'B',GEN,JUDGE,p,d,b.generate,b.judge,gate,{'revision':'fixed'})
            with self.assertRaisesRegex(RuntimeError,'interruption'): run_target(*args)
            self.assertEqual(len(b.gcalls),1)
            b.fail_at=None; r=run_target(*args); self.assertEqual(r['accepted_n'],2)
            self.assertEqual(len(b.gcalls),1); self.assertEqual(b.jcalls,5)
            before=(len(b.gcalls),b.jcalls); self.assertEqual(run_target(*args),r)
            self.assertEqual(before,(len(b.gcalls),b.jcalls))
    def test_binding_drift_rejected(self):
        b=Backend([['accepted']]); p=RetryPolicy(slots=1)
        with tempfile.TemporaryDirectory() as d:
            args=(SOURCE,'B',GEN,JUDGE,p,d,b.generate,b.judge,gate)
            run_target(*args,{'revision':'fixed'})
            with self.assertRaisesRegex(ValueError,'drift'): run_target(*args,{'revision':'changed'})
    def test_corrupt_checkpoint_rejected(self):
        b=Backend([['accepted']]); p=RetryPolicy(slots=1)
        with tempfile.TemporaryDirectory() as d:
            args=(SOURCE,'B',GEN,JUDGE,p,d,b.generate,b.judge,gate,{})
            run_target(*args); path=Path(d)/(sha('generate:unit-1:B:r0')+'.json')
            r=json.loads(path.read_text()); r['data'][0]['sentence']='tampered'; path.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'corruption'): run_target(*args)
    def test_invalid_policy_and_target(self):
        with self.assertRaises(ValueError): RetryPolicy(extra_rounds=3)
        with self.assertRaises(ValueError): RetryPolicy(slots=0)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): run_target(SOURCE,'A',GEN,JUDGE,RetryPolicy(),d,None,None,gate,{})
    def test_wrong_batch_count(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'batch size'):
                run_target(SOURCE,'B',GEN,JUDGE,RetryPolicy(slots=2),d,lambda *_:[],None,gate,{})
    def test_shared_pool_and_selector_integration(self):
        r,b=self.run_case([['cue','accepted','content'],['accepted']])
        pool=build_scoring_pool([SOURCE],[r],insert_before_final_sentence,sample_by_hash)
        self.assertEqual(len(pool['score_requests']),4)
        ids={c['candidate_id'] for c in pool['valid_candidates']}
        self.assertIn(pool['random_selection']['unit-1'],ids)
        scored=[dict(c,option_logprobs={'A':-3,'B':-1-i,'C':-5,'D':-6}) for i,c in enumerate(pool['valid_candidates'])]
        decision,_=decide_candidate_with_residual_policy(scored,'A','A','unit-1')
        self.assertIn(decision.selected_candidate_id,ids)
        for q in pool['score_requests'][1:]:
            self.assertEqual(q['options'],SOURCE['options']); self.assertTrue(q['question'].endswith('Which answer is best?'))
        with self.assertRaises(ValueError): build_scoring_pool([SOURCE],[r,r],insert_before_final_sentence,sample_by_hash)
    def test_empty_pool_preserves_clean(self):
        r,b=self.run_case([['cue']],RetryPolicy(slots=1,extra_rounds=0)); pool=build_scoring_pool([SOURCE],[r],insert_before_final_sentence,sample_by_hash)
        self.assertEqual(pool['source_coverage_n'],0); self.assertEqual(len(pool['score_requests']),1)
    def test_score_not_an_input(self):
        import inspect
        self.assertNotIn('score',inspect.signature(run_target).parameters)
        p=[dict(candidate_id='a',score=9),dict(candidate_id='b',score=-9)]
        self.assertEqual(sample_by_hash(p,'s','salt')['candidate_id'],sample_by_hash(list(reversed(p)),'s','salt')['candidate_id'])
    def test_no_speech_prefix_gate(self):
        candidate=dict(sentence='I joked about a printer.',finish_reason='stop')
        decision=classify_failure(candidate,dict(forward=judgment(),reverse=judgment()),'B','A',gate)
        self.assertEqual(decision['category'],'accepted')
    def test_actual_bystander_gate_identity(self):
        from common.hard_data.judge import target_gate_result
        self.assertIs(gate,target_gate_result)
        kinds=['accepted','cue','none','gold','multi','relation_none']
        for left in kinds:
            for right in kinds:
                js=dict(forward=judgment(left),reverse=judgment(right))
                result=classify_failure(dict(sentence='Text.',finish_reason='stop'),js,'B','A',gate)
                self.assertEqual(result['category']=='accepted',target_gate_result(js,'B','A')['pass'])
    def test_relation_none_follows_bystander_rule(self):
        r,b=self.run_case([['relation_none'],['accepted']]); self.assertEqual(len(b.gcalls),2)
    def test_judge_schema_excludes_nonliteral_content_fields(self):
        self.assertNotIn('usage_type',JUDGE_PROMPT)
        self.assertNotIn('patient_clinical_relevance',JUDGE_PROMPT)
        parsed,error=parse_judgment(json.dumps(judgment()['parsed']))
        self.assertIsNone(error); self.assertEqual(set(parsed),{'primary_evoked_option','evoked_options','relation_type','strength'})


if __name__=='__main__': unittest.main(verbosity=2)
