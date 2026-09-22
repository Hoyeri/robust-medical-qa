"""Small random Llama tests; no downloaded weights or clinical data."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from transformers.models.llama import modeling_llama
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from robust_medical_qa.attention import receiver_edges, receiver_mask, receivers
from robust_medical_qa.representations import Capture, transmission_features
from robust_medical_qa.runtime import ModelShape, label_tokens, load_runtime
from robust_medical_qa.io import read_json
from robust_medical_qa.extraction import extract
from experiments.attention_blocking import run
from experiments.attribution_probe import run as run_probe
from experiments.transmission_audit import analyze

ROOT = Path(__file__).resolve().parents[1]


def tiny_model():
    torch.manual_seed(818)
    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        attention_dropout=0., bos_token_id=1, eos_token_id=2, pad_token_id=0)
    cfg._attn_implementation = 'eager'
    return LlamaForCausalLM(cfg).eval()


def tiny_tokenizer():
    words = ['[UNK]','[BOS]','[EOS]','A','B','C','D','"','{','}',':',',','rationale','answer',
             'Base','patient','Her','coworker','also','has','signal','red','The','A','classmate','of','hers','.','blue']
    vocab = {word:i for i,word in enumerate(dict.fromkeys(words))}
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',bos_token='[BOS]',eos_token='[EOS]')
    tok.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    return tok


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = tiny_model()
        self.ids = [1,5,6,7,8,9]

    def test_mask_and_normalization(self):
        mask = torch.triu(torch.full((1,1,6,6),float('-inf')),diagonal=1)
        blocked = receiver_mask(mask,[3,4,5],[1,2])
        self.assertFalse(torch.equal(mask,blocked))
        weights = blocked.softmax(-1)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.allclose(weights.sum(-1),torch.ones(1,1,6)))
        self.assertEqual(weights[:,:,3:,[1,2]].count_nonzero().item(),0)

    def test_dynamic_shape_and_source_isolation(self):
        self.assertEqual(ModelShape.from_model(self.model).layers,2)
        def logits(ids, source):
            with torch.no_grad(), receiver_edges(self.model,[0,1],[3,4,5],source):
                return self.model(torch.tensor([ids]),use_cache=False).logits[0,-1]
        changed = self.ids.copy();changed[1:3]=[11,12]
        self.assertTrue(torch.equal(logits(self.ids,[1,2]),logits(changed,[1,2])))
        with torch.no_grad():
            base = self.model(torch.tensor([self.ids]),use_cache=False).logits
            with receiver_edges(self.model,[0,1],[],[1,2]):
                identity = self.model(torch.tensor([self.ids]),use_cache=False).logits
        self.assertTrue(torch.equal(base,identity))

    def test_exception_cleanup(self):
        original = modeling_llama.eager_attention_forward
        with self.assertRaises(RuntimeError):
            with receiver_edges(self.model,[0],[],[1]):
                raise RuntimeError('test interruption')
        self.assertIs(original,modeling_llama.eager_attention_forward)
        with self.assertRaises(RuntimeError):
            with Capture(self.model):
                raise RuntimeError('test interruption')
        self.assertIs(original,modeling_llama.eager_attention_forward)
        self.assertTrue(all(not b.self_attn.o_proj._forward_hooks for b in self.model.model.layers))
        self.assertFalse(self.model.model.embed_tokens._forward_hooks)

    def test_contributions_and_gradients(self):
        with torch.no_grad(), Capture(self.model) as cap:
            self.model(torch.tensor([self.ids]),use_cache=False)
            for layer in range(2):
                _, projected = cap.contribution(layer,list(range(len(self.ids))),[len(self.ids)-1])
                torch.testing.assert_close(projected[0],cap.oproj[layer][0,-1],atol=1e-7,rtol=1e-5)
        features,record,validation = transmission_features(self.model,self.ids,[1,2],
            {'sent':[1,2],'phr':[1,2]}, {'A':3,'B':4},'A','B',[0,1],[0,1],[.001])
        self.assertEqual(features['v'].shape,(2,16))
        self.assertEqual(features['h'].shape,(3,32))
        for row in validation:
            self.assertAlmostEqual(row['dq_actual'],row['dq_first_order'],delta=2e-7)
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        # Validation layer order must not change contributions captured from the baseline.
        _,_,reverse = transmission_features(self.model,self.ids,[1,2],{'sent':[1,2],'phr':[1,2]},
            {'A':3,'B':4},'A','B',[0,1],[1,0],[.001])
        self.assertEqual({r['layer']:r for r in validation},{r['layer']:r for r in reverse})

    def test_local_loader_and_end_to_end(self):
        tok = tiny_tokenizer()
        prompt = read_json(ROOT/'configs/prompt.json')
        _,labels = label_tokens(tok,prompt)
        self.assertNotEqual(labels['A'],32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.model.save_pretrained(root/'model');tok.save_pretrained(root/'model')
            model,tokenizer = load_runtime(root/'model')
            payload = {'generation':{'max_new_tokens':3,'stop_token_ids':[2],'use_cache':False}, 'rows':[
                {'question_id':'toy','gold_answer':'A','intended_target':'B','prompt_ids':self.ids,'options':[4],
                 'sources':{'distractor':[1],'clinical':[2]}}]}
            run(model,tokenizer,payload,read_json(ROOT/'configs/attention_blocking.json'),prompt,root/'attention')
            summary=read_json(root/'attention/summary.json')
            self.assertEqual(len(summary),5)
            self.assertTrue(all(r['n']==1 for r in summary.values()))
            items=[]
            for i in range(6):
                family='Her coworker' if i<3 else 'A classmate of hers'
                items.append({'item_id':f'toy{i}','gold_G':'A','target_T':'B','options':dict(A='red',B='blue',C='red red',D='blue blue'),
                    'variants':{'C':'Base.','Ma':'Base. The patient also has signal red.',
                                'H1':f'Base. {family} also has signal red.'},'meta':{'cue':{'hpo':f'concept{i}'}}})
            import json
            (root/'items.jsonl').write_text('\n'.join(json.dumps(it) for it in items))
            hidden={'variants':['Ma','H1'],'pooling_dtype':'native','storage_dtype':'float32'}
            extract(model,tokenizer,root/'items.jsonl',root/'hidden',hidden,prompt,'hidden')
            probe={'variants':['Ma','H1'],'indices':{'phrase':[0,1,2],'answer':[1,2]},
                   'probe':{'components':2,'folds':2,'seeds':[0]}}
            report=run_probe(root/'hidden',probe,items)
            self.assertEqual(report['independent_pairs'],6)
            transmission={'variants':['Ma','H1'],'storage_dtype':'float32','head_layers':[0,1],
                'validation_layers':[0,1],'validation_alphas':[.01],'control_variant':'Ma'}
            extract(model,tokenizer,root/'items.jsonl',root/'transmission',transmission,prompt,'transmission',validate=1)
            analysis=read_json(ROOT/'configs/transmission_analysis.json')
            analysis.update(indices={s:[0,1] for s in ['h','v','av','oav']},summary_indices=[0,1],
                            sensitivity_layers=[0,1],bootstrap={'repeats':40,'seed':1},contrasts=[['Ma','H1']])
            analysis['probe'].update(components=2,folds=2,seeds=[0])
            result=analyze(root/'transmission',items,analysis)
            self.assertEqual(result['independent_pairs'],6)
            self.assertEqual(len(result['stage_summary']),4)
            self.assertEqual(len(result['validation']),2)


if __name__ == '__main__':
    unittest.main()
