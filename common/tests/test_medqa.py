"""HF schema, legacy-ID alignment, and automatic construction inputs."""
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import io
import tempfile
import unittest

from common.checkpoints import atomic, jsonl, read
from common.hard_data.runtime import read_jsonl
from common.hard_data.scoring import DEFAULT_REVISION
from common.medqa import construction_order, load_medqa, match_medqa, medqa_source, normalize_rows
from common.tests.test_pipelines import SyntheticBackend
from pipeline.build_dataset import generate_pool, run as build
from pipeline.evaluate_models import run as evaluate


def hf_row(idx, question):
    return dict(id=f'test-{idx:05d}', sent1=question, sent2='', ending0='gold',
                ending1='topic_B', ending2='topic_C', ending3='topic_D', label=0)


class MedQATests(unittest.TestCase):
    def test_mapping_uses_content_and_retains_pair_ids_order(self):
        raw = [hf_row(228, 'One?'), hf_row(51, 'Two?')]
        rows = normalize_rows(raw, 'test')
        pairs = [dict(question_id=f'test-{i:05d}', source_idx=i, clean_question=r['question'],
                      distracted_question='Extra. '+r['question'], added_distractor='Extra.',
                      options=r['options'], gold_answer='A', intended_target='B',
                      selection_pool='harmful_flip_priority') for i, r in enumerate(rows)]
        matched, mapping = match_medqa(rows[::-1], pairs[::-1])
        self.assertEqual(matched, pairs[::-1])
        self.assertEqual(mapping, [dict(question_id='test-00001',hf_id='test-00051'),
                                   dict(question_id='test-00000',hf_id='test-00228')])
        for change in (dict(clean_question='Different?'), dict(gold_answer='C'),
                       dict(options=dict(pairs[0]['options'],A='wrong option'))):
            with self.assertRaisesRegex(ValueError,'No exact MedQA'):
                match_medqa(rows,[dict(pairs[0],**change)])
        with self.assertRaisesRegex(ValueError,'Duplicate Hard'):
            match_medqa(rows,[pairs[0],dict(pairs[0],question_id='other')])

    def test_schema_and_ambiguous_content(self):
        row=hf_row(0,'Question?')
        for change in (dict(label=True),dict(label=4),dict(label='A'),dict(sent2='Extra question'),
                       dict(ending0=''),dict(sent1=None),dict(id='dev-00000')):
            with self.subTest(change=change),self.assertRaises(ValueError):
                normalize_rows([dict(row,**change)],'test')
        with self.assertRaisesRegex(ValueError,'Duplicate MedQA question'):
            normalize_rows([row,dict(row,id='test-00001')],'test')
        # Repeated stems with different options are legitimate distinct MedQA items.
        self.assertEqual(len(normalize_rows([row,dict(row,id='test-00001',ending1='other')],'test')),2)

    def test_pinned_download_and_test_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.json';order_path=Path(tmp)/'order.json'
            jsonl(path,[hf_row(228,'One?'),hf_row(51,'Two?')])
            order=[dict(idx=1,hf_id='test-00228'),dict(idx=0,hf_id='test-00051')]
            atomic(order_path,order)
            with patch('huggingface_hub.hf_hub_download',return_value=str(path)) as download, \
                 patch('common.medqa.TEST_ORDER',order_path):
                rows,meta=load_medqa('test',construction=True)
            source=medqa_source('test')
            download.assert_called_once_with(repo_id=source['repo_id'],repo_type='dataset',
                filename=source['filename'],revision=source['revision'],token=False)
            self.assertEqual([r['source_id'] for r in rows],['test-00001','test-00000'])
            self.assertEqual([r['question'] for r in rows],['One?','Two?'])
            self.assertEqual(meta['alignment'][0],dict(source_id='test-00001',hf_id='test-00228'))
            with self.assertRaisesRegex(ValueError,'mapping'):
                construction_order(rows,[dict(idx=1,hf_id='test-00001')]*2)

    def test_hf_construction_matches_local_inputs_and_resumes_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            sources=normalize_rows([hf_row(4,'Patient arrived. What diagnosis?')],'test')
            jsonl(root/'local.jsonl',sources)
            for family in ('bystander','nonliteral'):
                def process(command,**kwargs):
                    if '--worker' in command:
                        state=Path(command[-1])
                        generate_pool(read(state/'sources.json'),read(state/'config.json'),state,SyntheticBackend())
                    else:
                        arg=lambda flag:Path(command[command.index(flag)+1])
                        records=[]
                        for req in read_jsonl(arg('--requests')):
                            target=req['intended_target'] or 'A'
                            scores={c:-4.0 for c in 'ABCD'};scores[target]=-1.0
                            records.append(dict(req,status='ok',model_revision=DEFAULT_REVISION,
                                                option_logprobs=scores,top1=target))
                        jsonl(arg('--output'),records);atomic(arg('--manifest'),dict(status='complete'))
                outputs=[]
                for mode in ('local','hf'):
                    args=SimpleNamespace(input=root/'local.jsonl' if mode=='local' else None,
                        split='test' if mode=='local' else None,type=family,retry_rounds=2,
                        output=root/(family+'_'+mode),resume=False)
                    with patch('pipeline.build_dataset.load_medqa',return_value=(sources,medqa_source('test'))) as download, \
                         patch('pipeline.build_dataset.subprocess.run',side_effect=process),redirect_stdout(io.StringIO()):
                        build(args)
                    if mode=='hf': download.assert_called_once_with('test',construction=True)
                    else: download.assert_not_called()
                    outputs.append({p.name:p.read_bytes() for p in args.output.glob('*') if p.is_file()})
                    args.resume=True
                    with patch('pipeline.build_dataset.load_medqa',side_effect=AssertionError('Repeated download')), \
                         patch('pipeline.build_dataset.subprocess.run',side_effect=AssertionError('Repeated inference')),redirect_stdout(io.StringIO()):
                        build(args)
                self.assertEqual(outputs[0],outputs[1])

    def test_mismatch_stops_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            row=normalize_rows([hf_row(0,'Known?')],'test')[0]
            pair=dict(question_id='old',source_idx=0,clean_question='Unknown?',
                      distracted_question='Extra. Unknown?',options=row['options'],gold_answer='A',
                      intended_target='B',added_distractor='Extra.',selection_pool='harmful_flip_priority')
            jsonl(root/'hard.jsonl',[pair])
            args=SimpleNamespace(input=root/'hard.jsonl',official=None,model='llama31_8b_instruct',
                                 output=root/'out',resume=False)
            with patch('pipeline.evaluate_models.load_medqa',return_value=([row],medqa_source('test'))), \
                 patch('pipeline.evaluate_models.subprocess.run') as inference:
                with self.assertRaisesRegex(ValueError,'No exact MedQA'):
                    evaluate(args)
                inference.assert_not_called()


if __name__ == '__main__':
    unittest.main()
