"""End-to-end orchestration, interruptions and shared family policy checks."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from common.checkpoints import Run, atomic, read, jsonl, sha, child_environment
from experiments.build_dataset import run as build, generate_pool, settings, sources_from
from common.hard_data.scoring import DEFAULT_REVISION
from common.hard_data.runtime import read_jsonl
from experiments.evaluate_models import paired_requests, prepare, match_official, result_table, format_results, official_source
from common.model_eval.protocol import analyze_pair_outputs
from common.tests.test_attention import tiny_model, tiny_tokenizer
from common.extraction import extract
from common.io import hard_dataset_filename, hard_dataset_path
from experiments.attention_blocking import run as attention

ROOT=Path(__file__).resolve().parents[2]


class SyntheticBackend:
    def __init__(self): self.target=None; self.calls=0
    def generate(self,prompt,n,seed):
        # A wrong-option topic is supplied to generation, labels are only test fixtures.
        self.target=next(t for t in 'BCD' if 'topic_'+t in prompt.split('## Clinical topic:')[-1])
        self.calls+=1
        return [dict(sentence=f'The unrelated topic_{self.target} example {i}.',finish_reason='stop',raw_generation='fixture',generated_token_ids=[1]) for i in range(n)]
    def judge(self,prompt):
        sentence=prompt.split('sentence:')[-1]
        target=self.target
        return dict(parsed=dict(primary_evoked_option=target,evoked_options=[target],relation_type='SEMANTIC',strength='strong'),
                    finish_reason='stop',error=None,raw='fixture',tokens=[])


class PipelineTests(unittest.TestCase):
    def test_source_validation_before_generation(self):
        source = dict(question='Patient arrived. What diagnosis?', options=dict(A='gold', B='b', C='c', D='d'), answer_idx='A')
        cases = [None, {}, dict(source, answer_idx=''), dict(source, answer_idx='AB'),
                 dict(source, answer_idx=None), dict(source, options=list('ABCD')), dict(source, question=' '),
                 dict(source, options=dict(A=None, B='b', C='c', D='d')),
                 dict(source, options=dict(A=' ', B='b', C='c', D='d'))]
        for row in cases:
            with self.subTest(row=row), self.assertRaises(ValueError):
                sources_from([row], 'test')
        self.assertEqual(sources_from([source], 'test'), [dict(source, idx=0, source_id='test-00000')])

    def test_build_both_families_through_selection_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=dict(idx=4,source_id='s4',question='Patient arrived. What diagnosis?',
                        options=dict(A='gold',B='topic_B',C='topic_C',D='topic_D'),answer_idx='A')
            jsonl(root/'input.jsonl',[source])
            for kind in ('bystander','nonliteral'):
                args=SimpleNamespace(input=root/'input.jsonl',split='internal',type=kind,retry_rounds=2,
                                     output=root/kind,resume=False)
                calls=[]
                def process(command,**kwargs):
                    calls.append(command)
                    if '--worker' in command:
                        state=Path(command[-1]);generate_pool(read(state/'sources.json'),read(state/'config.json'),state,SyntheticBackend())
                    else:
                        arg=lambda flag:Path(command[command.index(flag)+1])
                        records=[]
                        for req in read_jsonl(arg('--requests')):
                            target=req['intended_target'] or 'A'
                            scores={c:-4.0 for c in 'ABCD'};scores[target]=-1.0
                            records.append(dict(req,status='ok',model_revision=DEFAULT_REVISION,option_logprobs=scores,top1=target))
                        jsonl(arg('--output'),records);atomic(arg('--manifest'),dict(status='complete'))
                    return SimpleNamespace(returncode=0)
                with patch('experiments.build_dataset.subprocess.run',side_effect=process):build(args)
                self.assertEqual(len(calls),3)  # generation + direct score + independent repeat
                dataset=Path(args.output)/hard_dataset_filename(kind)
                self.assertEqual(hard_dataset_path(args.output),dataset)
                pairs=read_jsonl(dataset)
                self.assertEqual(len(pairs),1);self.assertIn(pairs[0]['intended_target'],'BCD')
                self.assertEqual(len(paired_requests(pairs)),2)
                self.assertEqual(read(Path(args.output)/'summary.json')['accepted_candidates'],24)
                args.resume=True
                with patch('experiments.build_dataset.subprocess.run',side_effect=AssertionError('Repeated inference')):build(args)
                self.assertEqual(pairs,read_jsonl(dataset))
                args.retry_rounds=0
                with self.assertRaisesRegex(ValueError,'drift'):build(args)

    def test_retry_configuration_is_shared_and_bounded(self):
        a=settings('bystander','internal');b=settings('nonliteral','internal')
        self.assertEqual({k:v for k,v in a['policy'].items() if k!='namespace'},
                         {k:v for k,v in b['policy'].items() if k!='namespace'})
        self.assertEqual(a['policy']['extra_rounds'],2)
        self.assertEqual(a['policy']['slots'],8)

    def test_worker_entrypoint_replays_completed_candidates(self):
        source=dict(idx=4,source_id='s4',question='Patient arrived. What diagnosis?',
                    options=dict(A='gold',B='topic_B',C='topic_C',D='topic_D'),answer_idx='A')
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ('bystander','nonliteral'):
                state=Path(tmp)/kind
                config=settings(kind,'internal')
                atomic(state/'sources.json',[source]);atomic(state/'config.json',config)
                generate_pool([source],config,state,SyntheticBackend())
                names=('pool.json','target_results.jsonl','score_requests.jsonl')
                expected={name:(state/name).read_bytes() for name in names}
                subprocess.run([sys.executable,str(ROOT/'experiments/build_dataset.py'),'--worker',str(state)],
                               cwd=tmp,env=child_environment(),check=True,capture_output=True)
                self.assertEqual(expected,{name:(state/name).read_bytes() for name in names})

    def test_completed_stage_tamper_and_single_writer(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run'
            with Run(out,{'input':'a'}) as run:
                def action():atomic(out/'result.json',{'value':1});return [out/'result.json']
                run.stage('compute',action)
                with self.assertRaises(BlockingIOError):Run(out,{'input':'a'},True)
            with Run(out,{'input':'a'},True) as run:run.stage('compute',lambda:1/0)
            atomic(out/'result.json',{'value':2})
            with Run(out,{'input':'a'},True) as run:
                with self.assertRaisesRegex(ValueError,'changed'):run.stage('compute',lambda:1/0)

    def test_extraction_resume_skips_completed_forwards(self):
        import torch
        torch.set_num_threads(1)
        model,tok=tiny_model(),tiny_tokenizer()
        prompt=read(ROOT/'common/prompt.json')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'items.jsonl'
            items=[dict(item_id='item',gold_G='A',target_T='B',options=dict(A='red',B='blue',C='red red',D='blue blue'),
                   variants=dict(C='Base.',Ma='Base. The patient also has signal red.',H1='Base. Her coworker also has signal red.'))]
            jsonl(path,items)
            cfg=dict(variants=['Ma','H1'],pooling_dtype='native',storage_dtype='float32')
            extract(model,tok,path,root/'out',cfg,prompt,'hidden')
            original={v:dict(np.load(root/'out'/f'{v}.npz')) for v in cfg['variants']}
            with patch('common.extraction.hidden_features',side_effect=AssertionError('Repeated forward')):
                extract(model,tok,path,root/'out',cfg,prompt,'hidden',resume=True)
            for v in cfg['variants']:
                for k,array in original[v].items():np.testing.assert_array_equal(array,np.load(root/'out'/f'{v}.npz')[k])
            cfg['pooling_dtype']='float32'
            with self.assertRaisesRegex(ValueError,'changed'):extract(model,tok,path,root/'out',cfg,prompt,'hidden',resume=True)

    def test_attention_resume_skips_completed_generation(self):
        import torch
        torch.set_num_threads(1)
        model,tok=tiny_model(),tiny_tokenizer();prompt=read(ROOT/'common/prompt.json')
        payload=dict(generation=dict(max_new_tokens=2,stop_token_ids=[2],use_cache=False),rows=[dict(
            question_id='item',gold_answer='A',intended_target='B',prompt_ids=[1,3,4,5,6],options=[4],sources=dict(distractor=[1],clinical=[2]))])
        cfg=read(ROOT/'common/configs/attention_blocking.json')
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out';attention(model,tok,payload,cfg,prompt,out)
            expected=read(out/'summary.json')
            with patch('experiments.attention_blocking.generate',side_effect=AssertionError('Repeated generation')):
                attention(model,tok,payload,cfg,prompt,out,resume=True)
            self.assertEqual(expected,read(out/'summary.json'))

    def test_two_view_summary_denominators(self):
        pair=dict(question_id='x',source_idx=1,clean_question='Clean?',distracted_question='Extra. Clean?',
                  added_distractor='Extra.',options=dict(A='a',B='b',C='c',D='d'),gold_answer='A',intended_target='B',selection_pool='harmful_flip_priority')
        requests=paired_requests([pair])
        rows=[dict(r,status='ok',generated_text=json.dumps(dict(rationale='test',answer='A' if r['view']=='clean' else 'B'))) for r in requests]
        result,_=analyze_pair_outputs(rows,expected_questions=1)
        self.assertEqual(result['own_clean_correct']['denominator'],1)
        self.assertEqual(result['own_clean_correct']['hard']['T']['count'],1)
        with self.assertRaises(ValueError):analyze_pair_outputs(rows+rows,expected_questions=1)



class EvaluationPipelineTests(unittest.TestCase):
    def test_direct_two_views_then_resume(self):
        self.check_evaluation(family=None, calibrated=False)

    def test_direct_official_views_then_resume(self):
        self.check_evaluation(family='bystander', calibrated=False)

    def test_direct_nonliteral_views_then_resume(self):
        self.check_evaluation(family='nonliteral', calibrated=False)

    def test_calibration_then_full_then_resume(self):
        self.check_evaluation(family=None)

    def test_official_three_views_then_resume(self):
        self.check_evaluation(family='bystander')

    def test_nonliteral_three_views_then_resume(self):
        self.check_evaluation(family='nonliteral')

    def check_evaluation(self, family, calibrated=True):
        from experiments.evaluate_models import run as evaluate
        from common.model_eval.protocol import parse_compatible_response, scientific_role_for_view, canonical_sha256
        from common.hard_data.runtime import sha256_file, write_json, write_jsonl
        real_run=subprocess.run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def pair(i,pool):
                return dict(question_id=f'i{i}',source_idx=i,clean_question=f'Case {i}?',
                    distracted_question=f'Extra. Case {i}?',options=dict(A='a',B='b',C='c',D='d'),gold_answer='A',
                    intended_target='B',added_distractor='Extra.',selection_pool=pool)
            dev=[pair(i,'harmful_flip_priority' if i<16 else 'maximum_D_fallback') for i in range(32)]
            jsonl(root/'dev.jsonl',dev);jsonl(root/'pairs.jsonl',[pair(100,'harmful_flip_priority')])
            hf_file=root/'official.json'
            if family:
                atomic(hf_file,[dict(
                    question='Extra. Another case?',distracting_sentence='Extra.',
                    question_choices=dict(A='a',B='b',C='c',D='d'),correct_answer='A'),dict(
                    question='Official extra. Case 100?',distracting_sentence='Official extra.',
                    question_choices=dict(A='a',B='b',C='c',D='d'),correct_answer='A')])
            if not calibrated:
                jsonl(root/'dataset'/hard_dataset_filename(family or 'bystander'),read_jsonl(root/'pairs.jsonl'))
            args=SimpleNamespace(input=root/'pairs.jsonl' if calibrated else root/'dataset',
                                 calibration=root/'dev.jsonl' if calibrated else None,official=family,
                                 model='llama31_8b_instruct',output=root/'out',resume=False)
            calls=[]
            def process(command,**kwargs):
                if command[2]!='common.model_eval.run':return real_run(command,**kwargs)
                get=lambda flag:command[command.index(flag)+1]
                mode=get('--mode');calls.append(mode);path=Path(get('--output-dir'));path.mkdir(parents=True,exist_ok=True)
                if mode == 'full':
                    self.assertEqual('--calibration-gate' in command, calibrated)
                requests_path=Path(get('--requests'));protocol_path=Path(get('--protocol'))
                requests=read_jsonl(requests_path);rows=[]
                for r in requests:
                    text=json.dumps(dict(rationale='test',answer='A' if r['view']=='clean' else 'B'))
                    parsed=parse_compatible_response(text)
                    rows.append(dict(r,status='ok',model_key='llama31_8b_instruct',track_key='general_models_three_view_rationale_1024',
                        max_new_tokens=1024,protocol_sha256=sha256_file(protocol_path),request_sha256=canonical_sha256(r),
                        generated_text=text,**parsed,generated_token_count=10,input_token_count=10,
                        finish_reason='stop',max_token_cutoff=False,
                        scientific_role=scientific_role_for_view(parsed['pred_answer'],view=r['view'],gold=r['gold_answer'],hard_target=r['hard_intended_target'])))
                write_jsonl(path/'outputs.jsonl',rows)
                write_json(path/'run_manifest.json',dict(status='complete',model_key='llama31_8b_instruct',mode=mode,
                    track_key='general_models_three_view_rationale_1024',max_new_tokens=1024,
                    protocol_sha256=sha256_file(protocol_path),bindings={
                        'requests':dict(sha256=sha256_file(requests_path)),
                        'outputs':dict(sha256=sha256_file(path/'outputs.jsonl'))}))
                return SimpleNamespace(returncode=0)
            terminal=io.StringIO()
            with patch('huggingface_hub.hf_hub_download',return_value=str(hf_file)) as download, \
                 patch('experiments.evaluate_models.subprocess.run',side_effect=process),redirect_stdout(terminal):
                evaluate(args)
            if family:
                source=official_source(family)
                download.assert_called_once_with(repo_id=source['repo_id'],repo_type='dataset',
                    filename=source['filename'],revision=source['revision'],token=False)
            else:
                download.assert_not_called()
            self.assertEqual(calls,['calibration','full'] if calibrated else ['full'])
            self.assertEqual((root/'out/.state/calibration').exists(),calibrated)
            self.assertEqual(read(root/'out/summary.json')['results']['questions'],1)
            predictions=read_jsonl(root/'out/predictions.jsonl')
            expected_views={'clean','hard','meddistractqa'} if family else {'clean','hard'}
            self.assertEqual({r['view'] for r in predictions},expected_views)
            self.assertEqual(len(predictions),len(expected_views))
            summary=read(root/'out/summary.json')
            self.assertEqual((root/'out/summary.txt').read_text(),format_results(summary))
            self.assertIn(format_results(summary),terminal.getvalue())
            with (root/'out/summary.csv').open() as stream: table=list(csv.DictReader(stream))
            self.assertEqual({r['condition'] for r in table},expected_views)
            self.assertEqual([float(r['accuracy']) for r in table],[r['accuracy'] for r in result_table(summary)])
            if family:
                official_prediction=next(r for r in predictions if r['view']=='meddistractqa')
                self.assertIsNone(official_prediction['intended_target'])
                self.assertNotEqual(official_prediction['scientific_role'],'T')
                self.assertEqual(summary['official']['matched_questions'],1)
                self.assertEqual(summary['official']['available_questions'],2)
                self.assertNotIn('revision',summary['official'])
            args.resume=True
            with patch('experiments.evaluate_models.subprocess.run',side_effect=AssertionError('Repeated inference')), \
                 patch('experiments.evaluate_models.load_official',side_effect=AssertionError('Repeated download')),redirect_stdout(io.StringIO()):
                evaluate(args)

    def test_worker_keeps_gate_for_original_protocol(self):
        from common.model_eval.run import main as generate
        from common.model_eval.protocol import EXPECTED_PROTOCOL_STATUS, DIRECT_PROTOCOL_STATUS
        pair = dict(question_id='x', source_idx=1, clean_question='Case?', distracted_question='Extra. Case?',
                    added_distractor='Extra.', options=dict(A='a', B='b', C='c', D='d'), gold_answer='A',
                    intended_target='B', selection_pool='harmful_flip_priority')
        with tempfile.TemporaryDirectory() as d:
            state=Path(d)
            registry=read(ROOT/'common/model_eval/models.json')
            prepare([pair],None,None,registry,'llama31_8b_instruct',state)
            protocol=read(state/'protocol.json')
            self.assertEqual(protocol['status'],DIRECT_PROTOCOL_STATUS)
            argv=['run','--protocol',str(state/'protocol.json'),'--requests',str(state/'full.jsonl'),
                  '--model-key','llama31_8b_instruct','--mode','full','--output-dir',str(state/'generation')]
            with patch.object(sys,'argv',argv), patch('common.model_eval.run.importlib.metadata.version',
                    side_effect=RuntimeError('Reached inference runtime')) as runtime:
                with self.assertRaisesRegex(RuntimeError,'Reached inference runtime'):
                    generate()
                runtime.assert_called_once()
            protocol['status']=EXPECTED_PROTOCOL_STATUS
            atomic(state/'protocol.json',protocol)
            with patch.object(sys,'argv',argv), patch('common.model_eval.run.importlib.metadata.version') as runtime:
                with self.assertRaisesRegex(ValueError,'calibration-gate is required'):
                    generate()
                runtime.assert_not_called()

    def test_official_matching_reorders_and_checks_content(self):
        pair=dict(question_id='local-id',source_idx=999,clean_question='Case one? ',
            distracted_question='Extra. Case one?',added_distractor='Extra.',options=dict(A='a',B='b',C='c',D='d'),
            gold_answer='A',intended_target='B',selection_pool='harmful_flip_priority')
        official=dict(question='  Official extra.  Case one?\n',distracting_sentence='Official extra.',
                      question_choices=pair['options'],correct_answer='A')
        extra=dict(official,question='Official extra. Case two?')
        for family in ('bystander','nonliteral'):
            matched,meta=match_official([extra,official],[pair],family)
            self.assertEqual(matched[0]['question'],official['question'])
            self.assertEqual(matched[0]['idx'],999)
            self.assertEqual(meta['alignment'],[dict(question_id='local-id',official_row=1)])
        cases=[([dict(official,question='Official extra. Other?')], 'No official'),
               ([official,official], 'duplicate'),
               ([dict(official,correct_answer='B')], 'gold mismatch'),
               ([dict(official,question_choices=dict(A='b',B='a',C='c',D='d'))], 'options mismatch'),
               ([dict(official,question='Official extra. Official extra. Case one?')], 'exactly once')]
        for rows,message in cases:
            with self.assertRaisesRegex(ValueError,message):match_official(rows,[pair],'bystander')
        with self.assertRaisesRegex(ValueError,'Duplicate Hard'):match_official([official],[pair,pair],'bystander')

    def test_result_denominators_include_invalid_and_cutoff(self):
        from common.model_eval.protocol import analyze_triad_outputs
        rows=[]
        for i in range(3):
            for view in ('clean','meddistractqa','hard'):
                answer='A' if view=='clean' or i==0 else 'B'
                text='unparseable' if i==2 and view!='clean' else json.dumps(dict(rationale='test',answer=answer))
                rows.append(dict(question_id=str(i),view=view,gold_answer='A',hard_intended_target='B',
                                 generated_text=text,max_token_cutoff=(i==2)))
        counts,_=analyze_triad_outputs(rows,expected_questions=3)
        table=result_table(dict(results=counts))
        for row in table[1:]:
            self.assertEqual((row['questions'],row['correct'],row['wrong'],row['invalid']),(3,1,1,1))
            self.assertEqual(row['max_token_cutoff'],1)
            self.assertAlmostEqual(row['accuracy'],1/3)
            self.assertAlmostEqual(row['accuracy_change_from_clean_pp'],-200/3)

    def test_official_schema_rejects_invalid_option_content(self):
        pair = dict(question_id='x', source_idx=0, clean_question='Case?', distracted_question='Hard. Case?',
                    added_distractor='Hard.', options=dict(A='a', B='b', C='c', D='d'), gold_answer='A',
                    intended_target='B', selection_pool='harmful_flip_priority')
        official = dict(question='Extra. Case?', distracting_sentence='Extra.', question_choices=pair['options'], correct_answer='A')
        for row in (None, {}, dict(official, question_choices=list('ABCD'))):
            with self.subTest(row=row), self.assertRaises(ValueError):
                match_official([row], [pair], 'bystander')
        for value in (None, 3, [], {}, '', ' '):
            choices = dict(pair['options'], A=value)
            # Equality between two malformed inputs must not count as a valid match.
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Invalid official options'):
                match_official([dict(official, question_choices=choices)], [dict(pair, options=choices)], 'bystander')


if __name__ == '__main__':
    unittest.main()
