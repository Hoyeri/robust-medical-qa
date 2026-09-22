"""Natural autoregressive generation under frozen attention-edge policies."""
import argparse,json,time
from run_e01 import ROOT,SOURCE,DEADLINE,LABELS,PAIR_SHA,data,require,digest,atomic,token_digest
from discovery_runtime import execution_guard,FrozenRuntime,utcnow
from run_e06 import PAYLOAD as E06_PAYLOAD,OUTPUT as E06_OUTPUT,CODE as E06_CODE,LAYERS
from receiver_knockout import receiver_edges,toy_tests
from generation_routes import select_ids,generation_queries,POLICIES

PROTOCOL=ROOT/'experiments/E08_NATURAL_GENERATION_V1_KO.md'
PAYLOAD=ROOT/'data/e08_natural_generation_v1.json'
OUTPUT=ROOT/'results/e08_natural_generation_v1'
CODE=E06_CODE+['generation_routes.py','run_e08.py']
CONFIG=SOURCE/'configs/rationale_generation_v1.json'
REUSE=['src/clean_hard_xai/generation_v1/decoding.py','src/clean_hard_xai/generation_v1/records.py']

def prepare():
    pairs=data();lookup={p['question_id']:p for p in pairs};selected=select_ids(list(lookup));e06={p['question_id']:p for p in json.loads(E06_PAYLOAD.read_text())['rows']}
    cfg=json.loads(CONFIG.read_text());require(cfg['max_new_tokens']==1024 and cfg['use_cache'] is False and cfg['stop_token_ids']==[128001,128008,128009] and cfg['assistant_prefill']=='' and cfg['json_constrained_decoding'] is False,'Original generation contract')
    rows=[]
    for qid in selected:
        p=lookup[qid];e=p['hard'];a=e06[qid];s=a['stages']['empty'];prompt=e['input_token_ids'][:e['prompt_token_n']]
        require(s['ids'][:s['prompt_n']]==prompt and s['prompt_n']==len(prompt),'Exact prompt binding')
        expected=e['input_token_ids'][len(prompt):]+[e['expected_trace']['token_id']]
        require(expected[-1]==LABELS[e['baseline_answer']] and len(expected)==e['generated_prefix_token_n']+1,'Expected prefix/answer')
        row={k:a[k] for k in ['question_id','group','gold_answer','intended_target']}
        row.update({'prompt_ids':prompt,'prompt_sha256':token_digest(prompt),'source':s['source'],'options':s['families']['options'],
          'expected_baseline_tokens_through_answer':expected,'expected_answer_trace':e['expected_trace'],'baseline_answer':e['baseline_answer'],
          'source_endpoint_sha256':digest(SOURCE/'data/clean_hard_patching_full_v1/pairs.json')})
        require(generation_queries('all_source_cut',len(s['ids']),row['source'],row['options'])==s['all_queries'],'E06 allreceiver contract')
        rows.append(row)
    value={'schema':'e08_data_v1','n':96,'selection':'96 smallest sha256(e08-natural-v1:+qid), no outcome selection','protocol_sha256':digest(PROTOCOL),'e06_payload_sha256':digest(E06_PAYLOAD),'pair_sha256':PAIR_SHA,'generation_config_sha256':digest(CONFIG),'cfg':cfg,'rows':rows}
    if PAYLOAD.exists():require(json.loads(PAYLOAD.read_text())==value,'Frozen E08 payload')
    else:atomic(PAYLOAD,value)
    atomic(ROOT/'evidence/e08_preparation.json',{'status':'pass','n':96,'ids':selected,'payload_sha256':digest(PAYLOAD),'expected_original_prefix_tokens':sum(len(r['expected_baseline_tokens_through_answer']) for r in rows)})
    print(json.dumps({'status':'prepared','n':96,'payload_sha256':digest(PAYLOAD)}),flush=True)

def main(args):
    if args.prepare_only:prepare();return
    payload=json.loads(PAYLOAD.read_text());data();require(digest(PROTOCOL)==payload['protocol_sha256'] and digest(E06_PAYLOAD)==payload['e06_payload_sha256'] and digest(CONFIG)==payload['generation_config_sha256'],'Input bindings')
    require(json.loads((E06_OUTPUT/'summary.json').read_text())['status']=='verified_complete','E06 verified')
    require([r['question_id'] for r in payload['rows']]==select_ids([p['question_id'] for p in data()]),'IDhash selection')
    old=json.loads((E06_OUTPUT/'contract.json').read_text());require(all(digest(ROOT/'scripts'/p)==sha for p,sha in old['code'].items()),'Frozen E06 code')
    contract={'schema':'e08_v1','n':96,'policies':POLICIES,'new_generations':288,'payload_sha256':digest(PAYLOAD),'protocol_sha256':digest(PROTOCOL),'deadline':DEADLINE,
      'code':{p:digest(ROOT/'scripts'/p) for p in CODE},'original_generation_sources':{p:digest(SOURCE/p) for p in REUSE},
      'e06_shards':{r['question_id']+'.json':digest(E06_OUTPUT/'shards'/(r['question_id']+'.json')) for r in payload['rows']}}
    with execution_guard(OUTPUT,contract):
        import torch
        toy_tests(torch);rt=FrozenRuntime(OUTPUT)
        from transformers import AutoTokenizer
        from clean_hard_xai.generation_v1.decoding import greedy_tokens,trace_logits
        from clean_hard_xai.generation_v1.records import parse,align_answer,replay_offsets
        import sys
        from pathlib import Path
        for module in list(sys.modules.values()):
            f=getattr(module,'__file__',None)
            if f and str(Path(f).resolve()).startswith(str(SOURCE.parent)+'/') and not str(Path(f).resolve()).startswith(str(ROOT)+'/'):rt.reused[str(Path(f).resolve())]=digest(f)
        atomic(OUTPUT/'reused_sources.json',rt.reused)
        tok=AutoTokenizer.from_pretrained(SOURCE/'results/stage0b_preflight_v1/resolved_tokenizer',local_files_only=True,trust_remote_code=False,use_fast=True)
        cfg=payload['cfg'];require(list(rt.model.generation_config.eos_token_id)==cfg['stop_token_ids'],'EOS binding')
        e06={r['question_id']:r for r in json.loads(E06_PAYLOAD.read_text())['rows']};start=time.monotonic();limit=3 if args.smoke_only else 96
        for i,row in enumerate(payload['rows'][:limit]):
            qid=row['question_id'];s=row['source'];op=row['options'];prompt=row['prompt_ids']
            require(token_digest(prompt)==row['prompt_sha256'],'Prompt hash')
            def forward(ids,policy):
                if policy=='baseline':return rt.forward(ids)
                qs=generation_queries(policy,len(ids),s,op)
                with receiver_edges(torch,LAYERS,qs,s):return rt.forward(ids)
            if i<3:
                anchors=json.loads((E06_OUTPUT/'shards'/(qid+'.json')).read_text())
                for stage in ['empty','full']:
                    ids=e06[qid]['stages'][stage]['ids'];a=anchors['results'][stage]
                    require(rt.score(forward(ids,'baseline'),row)==a['baseline'],'E06 baseline exact')
                    require(rt.score(forward(ids,'option_cut'),row)==a['conditions']['block_only__options']['score'],'E06 options exact')
                    require(rt.score(forward(ids,'all_source_cut'),row)==a['all_blocked'],'E06 allcut exact')
                raw=rt.forward(prompt)
                with receiver_edges(torch,LAYERS,[],s):identity=rt.forward(prompt)
                rt.equal(raw,identity,'Empty receiver fullvocab')
                modified=list(prompt)
                for j in s:modified[j]=1000+(j%2)
                rt.equal(forward(prompt,'all_source_cut'),forward(modified,'all_source_cut'),'Source-ID isolation exact')
                rt.equal(raw,rt.forward(prompt),'Hook removed')
            for policy in POLICIES:
                path=OUTPUT/'conditions'/(qid+'__'+policy+'.json')
                if path.exists():require(json.loads(path.read_text())['payload_sha256']==contract['payload_sha256'],'Resume binding');continue
                fn=lambda ids:forward(ids,policy)
                ids,trace=greedy_tokens(fn,torch,prompt,cfg,LABELS)
                text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False);parsed=parse(text,row);alignment=align_answer(tok,ids,text,parsed,len(prompt))
                if policy=='baseline':
                    expected=row['expected_baseline_tokens_through_answer'];j=len(expected)-1
                    require(ids[:len(expected)]==expected and trace[j]==row['expected_answer_trace'],'Fresh baseline original tokens and answertrace exact')
                    require(parsed['pred_answer']==row['baseline_answer'],'Fresh baseline answer exact')
                replays=[]
                for j in replay_offsets(len(ids),alignment):
                    fresh=trace_logits(fn(prompt+ids[:j]),torch,j,LABELS);require(fresh==trace[j],'Generation same-policy prefix replay exact')
                    replays.append({'offset':j,'trace':fresh,'prefix_sha256':token_digest(prompt+ids[:j]),'exact_equal':True})
                atomic(path,{**{k:row[k] for k in ['question_id','group','gold_answer','intended_target']},'policy':policy,'payload_sha256':contract['payload_sha256'],
                  'generated_token_ids':ids,'generated_token_ids_sha256':token_digest(ids),'generated_text':text,'token_trace':trace,'parsed':parsed,'answer_alignment':alignment,
                  'finish_reason':'stop' if ids[-1] in cfg['stop_token_ids'] else 'length','replays':replays,'baseline_original_exact':policy=='baseline','must_not_be_used_for_training':True})
                elapsed=time.monotonic()-start;atomic(OUTPUT/'progress.json',{'completed_conditions':len(list((OUTPUT/'conditions').glob('*.json'))),'planned_conditions':limit*3,'current_pair':i+1,'current_policy':policy,'elapsed_s':elapsed,'eta_s':elapsed/((i*3)+POLICIES.index(policy)+1)*(limit*3-((i*3)+POLICIES.index(policy)+1)),'pid':__import__('os').getpid()})
                print(json.dumps({'pair':i+1,'n':limit,'policy':policy,'tokens':len(ids),'parsed_role':parsed['scientific_role'],'elapsed_s':round(elapsed,2)}),flush=True)
            if i==2:atomic(OUTPUT/'smoke.json',{'status':'pass','n':3,'checks':['E06 bothstages baseline/options/allcut exact','empty/removal fullvocab exact','source-ID isolation fullvocab exact','fresh baseline tokens/answertrace exact','every policy generation prefix replay exact']})
        rt.recheck_sources();data()
        require(all(digest(ROOT/'scripts'/p)==sha for p,sha in contract['code'].items()) and all(digest(SOURCE/p)==sha for p,sha in contract['original_generation_sources'].items()) and digest(PAYLOAD)==contract['payload_sha256'],'Frozen sources unchanged')
        require(all(digest(E06_OUTPUT/'shards'/p)==sha for p,sha in contract['e06_shards'].items()),'Anchor unchanged')
        records=[json.loads(p.read_text()) for p in (OUTPUT/'conditions').glob('*.json')]
        atomic(OUTPUT/'completion.json',{'status':'smoke_complete' if args.smoke_only else 'complete','n':limit,'generations':len(records),'generated_tokens':sum(len(r['generated_token_ids']) for r in records),'replay_forward_n':sum(len(r['replays']) for r in records),'completed_at':utcnow(),'source_rechecked':True})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke-only',action='store_true');main(p.parse_args())
