"""Matched original-question source control for natural autoregressive generation."""
import argparse,json,time
from run_e01 import ROOT,SOURCE,DEADLINE,LABELS,data,require,digest,atomic,token_digest
from discovery_runtime import execution_guard,FrozenRuntime,utcnow
from run_e08 import PAYLOAD as E08_PAYLOAD,OUTPUT as E08_OUTPUT,CODE as E08_CODE,CONFIG,REUSE
from run_e05 import PAYLOAD as E05_PAYLOAD,OUTPUT as E05_OUTPUT
from receiver_knockout import receiver_edges,toy_tests
from generation_routes import generation_queries
from natural_trajectory import NaturalTrajectory

PROTOCOL=ROOT/'experiments/E11_NATURAL_SOURCE_CONTROL_V1_KO.md'
PAYLOAD=ROOT/'data/e11_natural_source_control_v1.json'
OUTPUT=ROOT/'results/e11_natural_source_control_v1'
POLICIES=['option_cut','all_source_cut'];LAYERS=list(range(32))
CODE=E08_CODE+['natural_trajectory.py','run_e11.py']

def prepare():
    e08=json.loads(E08_PAYLOAD.read_text());e05={r['question_id']:r for r in json.loads(E05_PAYLOAD.read_text())['rows']};allowed={r['question_id'] for r in data()};rows=[]
    for r in e08['rows']:
        q=r['question_id'];require(q in allowed,'Exploration ID');a=e05[q]['stages']['empty']['H'];s=a['spans']['clinical_matched']
        require(a['ids'][:len(r['prompt_ids'])]==r['prompt_ids'],'Same Hard prompt')
        require(s and len(s)==len(r['source']) and not set(s)&(set(r['source'])|set(r['options'])) and 0 not in s and max(s)<min(r['options']),'Matched original source contract')
        require(generation_queries('all_source_cut',len(a['ids']),s,r['options'])==a['queries']['clinical_matched']['downstream'],'Frozen E05 query contract')
        rows.append({**r,'distractor_source':r['source'],'source':s})
    require(len(rows)==96,'96 fixed selection')
    value={'schema':'e11_data_v1','protocol_sha256':digest(PROTOCOL),'e08_payload_sha256':digest(E08_PAYLOAD),'e05_payload_sha256':digest(E05_PAYLOAD),'cfg':e08['cfg'],'rows':rows}
    if PAYLOAD.exists():require(json.loads(PAYLOAD.read_text())==value,'Frozen payload')
    else:atomic(PAYLOAD,value)
    print(json.dumps({'status':'prepared','n':96,'new_generations':192,'payload_sha256':digest(PAYLOAD)}),flush=True)

def main(args):
    if args.prepare_only:prepare();return
    for name in ['e08_natural_generation_v1','e10_option_state_v1']:
        p=ROOT/'results'/name;require(json.loads((p/'process.json').read_text())['status']=='finished' and json.loads((p/'summary.json').read_text())['status']=='verified_complete','Predecessor complete verified')
    payload=json.loads(PAYLOAD.read_text());data();require(payload['protocol_sha256']==digest(PROTOCOL) and payload['e08_payload_sha256']==digest(E08_PAYLOAD) and payload['e05_payload_sha256']==digest(E05_PAYLOAD),'Bindings')
    old=json.loads((E08_OUTPUT/'contract.json').read_text());require(all(digest(ROOT/'scripts'/k)==v for k,v in old['code'].items()),'Frozen E08 runtime')
    contract={'schema':'e11_v1','n':96,'policies':POLICIES,'new_generations':192,'extra_baseline_smoke_generations':3,'payload_sha256':digest(PAYLOAD),'protocol_sha256':digest(PROTOCOL),'deadline':DEADLINE,
        'code':{k:digest(ROOT/'scripts'/k) for k in CODE},'original_generation_sources':{k:digest(SOURCE/k) for k in REUSE},'generation_config_sha256':digest(CONFIG),
        'e08_records':{p.name:digest(p) for p in (E08_OUTPUT/'conditions').glob('*.json')},'e05_shards':{r['question_id']:digest(E05_OUTPUT/'shards'/(r['question_id']+'.json')) for r in payload['rows']}}
    with execution_guard(OUTPUT,contract):
        import torch
        toy_tests(torch);rt=FrozenRuntime(OUTPUT);gen=NaturalTrajectory(rt,payload['cfg'],OUTPUT);e05={r['question_id']:r for r in json.loads(E05_PAYLOAD.read_text())['rows']};start=time.monotonic();limit=3 if args.smoke_only else 96
        for i,row in enumerate(payload['rows'][:limit]):
            qid=row['question_id'];prompt=row['prompt_ids'];s=row['source'];op=row['options'];require(token_digest(prompt)==row['prompt_sha256'],'Prompt hash')
            def forward(ids,policy):
                with receiver_edges(torch,LAYERS,generation_queries(policy,len(ids),s,op),s):return rt.forward(ids)
            if i<3:
                oldbase=json.loads((E08_OUTPUT/'conditions'/(qid+'__baseline.json')).read_text());fresh=gen.generate(prompt,row,rt.forward)
                require(all(fresh[k]==oldbase[k] for k in fresh),'New shared generation recorder exact E08')
                atomic(OUTPUT/'smoke_baselines'/(qid+'.json'),fresh)
                anchors=json.loads((E05_OUTPUT/'shards'/(qid+'.json')).read_text())
                for stage in ['empty','full']:
                    a=e05[qid]['stages'][stage]['H'];require(rt.score(forward(a['ids'],'all_source_cut'),row)==anchors['results'][stage]['H']['effects']['clinical_matched']['conditions']['all32__downstream']['score'],'E05 original source anchor exact')
                base=rt.forward(prompt)
                with receiver_edges(torch,LAYERS,[],s):identity=rt.forward(prompt)
                rt.equal(base,identity,'Empty receivers fullvocab exact');rt.equal(base,rt.forward(prompt),'Removal fullvocab exact')
                modified=list(prompt)
                for j in s:modified[j]=1000 if prompt[j]!=1000 else 1001
                rt.equal(forward(prompt,'all_source_cut'),forward(modified,'all_source_cut'),'Original source-ID isolation exact')
            for policy in POLICIES:
                path=OUTPUT/'conditions'/(qid+'__'+policy+'.json')
                if path.exists():require(json.loads(path.read_text())['payload_sha256']==contract['payload_sha256'],'Resume binding');continue
                record=gen.generate(prompt,row,lambda ids:forward(ids,policy))
                atomic(path,{**{k:row[k] for k in ['question_id','group','gold_answer','intended_target']},'policy':policy,'source_kind':'clinical_matched','payload_sha256':contract['payload_sha256'],**record})
                done=len(list((OUTPUT/'conditions').glob('*.json')));elapsed=time.monotonic()-start
                atomic(OUTPUT/'progress.json',{'completed_conditions':done,'planned_conditions':limit*2,'current_pair':i+1,'elapsed_s':elapsed,'eta_s':elapsed/done*(limit*2-done),'pid':__import__('os').getpid()})
                print(json.dumps({'pair':i+1,'n':limit,'policy':policy,'tokens':len(record['generated_token_ids']),'role':record['parsed']['scientific_role'],'elapsed_s':round(elapsed,2)}),flush=True)
            if i==2:atomic(OUTPUT/'smoke.json',{'status':'pass','n':3,'checks':['Fresh baseline all fields exact E08','E05 empty/full sourcecut exact','Empty/removal fullvocab exact','Source-ID isolation exact','Percondition prefix replay exact']})
        rt.recheck_sources();data();require(digest(PAYLOAD)==contract['payload_sha256'] and all(digest(ROOT/'scripts'/k)==v for k,v in contract['code'].items()),'Frozen code/data')
        require(all(digest(E08_OUTPUT/'conditions'/k)==v for k,v in contract['e08_records'].items()) and all(digest(E05_OUTPUT/'shards'/(k+'.json'))==v for k,v in contract['e05_shards'].items()),'Anchors unchanged')
        records=[json.loads(p.read_text()) for p in (OUTPUT/'conditions').glob('*.json')]
        atomic(OUTPUT/'completion.json',{'status':'smoke_complete' if args.smoke_only else 'complete','n':limit,'generations':len(records),'extra_baseline_smoke_generations':min(limit,3),
            'generated_tokens':sum(len(r['generated_token_ids']) for r in records),'replay_forward_n':sum(len(r['replays']) for r in records),'completed_at':utcnow(),'source_rechecked':True})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke-only',action='store_true');main(p.parse_args())
