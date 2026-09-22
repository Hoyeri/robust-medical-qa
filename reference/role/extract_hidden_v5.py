# -*- coding: utf-8 -*-
"""Step 1 data: hidden states for the contextual-role probe.

For each item and variant (Ma, H1, Oa, Pg, Ua) save, for all 33 hidden-state layers (0 = embeddings):
    phrase  mean hidden state over the FINDING-PHRASE tokens inside the inserted sentence (template words excluded,
            so for Ma vs H1 the strings are identical -> a layer-0 probe must be at chance by construction)
    answer  hidden state at the answer slot (last position of prompt + RS prefix)
Output: <out>/<variant>.npz with arrays phrase[n,33,4096], answer[n,33,4096] (float16) and item_ids.

Usage: CUDA_VISIBLE_DEVICES=3 python deps/extract_hidden_v5.py items/cm_confirmatory_v5.jsonl --out results/hidden_v5
"""
import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np
DEPS = Path('/data/yrhong/MedFuzz-Reimplementation/0920_cm_pilot_hamster/deps'); sys.path.insert(0, str(DEPS))
assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('2', '3'), 'pin to Hamster GPU2 or GPU3 (approved GPUs only)'
import ariel_gsweep_common as C
C.ROOT = DEPS; C.TOK_DIR = DEPS / 'tokenizer'
C.MODEL_SNAPSHOT = Path('/data/yrhong/huggingface_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659')
C.slurm_guard = lambda: None
import transformers
if int(transformers.__version__.split('.')[0]) >= 5:
    _orig = transformers.AutoModelForCausalLM.from_pretrained.__func__
    def _fp(cls, *a, **k):
        if 'torch_dtype' in k: k['dtype'] = k.pop('torch_dtype')
        k.pop('low_cpu_mem_usage', None)
        return _orig(cls, *a, **k)
    transformers.AutoModelForCausalLM.from_pretrained = classmethod(_fp)
import ariel_cm_harness_v3 as H3
_norm = lambda t: re.sub(r'\s+', ' ', str(t)).strip()
TEMPLATE = re.compile(r"^(?:(?:She|He|The patient|Her coworker|His coworker|The patient's coworker|A classmate of (?:hers|his|the patient's)|Another (?:baby|child) at the clinic|Her parents|His parents|The patient's parents) "
                      r"(?:also has|has a history of|has had|has|previously had|reports a past history of|also complains of|additionally reports|reports|complains of|also have)\s+"
                      r"|(?:Examination also reveals|Examination additionally shows|Physical examination also demonstrates|Laboratory studies also show|Laboratory testing additionally reveals|Laboratory results also demonstrate|Additional laboratory studies show|Imaging also shows|Imaging additionally reveals|Imaging studies also demonstrate|There is a history of)\s+"
                      r"|(?:On further questioning, (?:she|he|the patient) reports)\s+|(?:On examination, |On imaging, )"
                      r"|(?:She|He|The patient) (?:denies|has no history of)\s+|(?:Examination reveals no|Laboratory studies show no|Imaging shows no)\s+)")

def inserted_sentence(c_text, v_text):
    c, t = _norm(c_text), _norm(v_text); i = 0
    while i < len(c) and c[i] == t[i]: i += 1
    j = 0
    while j < len(c) - i and c[len(c) - 1 - j] == t[len(t) - 1 - j]: j += 1
    return t[i:len(t) - j].strip()

def phrase_span_in(sentence):
    """char range of the finding phrase inside the inserted sentence (template prefix stripped, trailing punctuation dropped)"""
    m = TEMPLATE.match(sentence); start = m.end() if m else 0
    body = sentence.rstrip('.').rstrip()
    body = re.sub(r'\s+(is also noted|is also seen)$', '', body)
    return start, len(body)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('items'); ap.add_argument('--out', required=True); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--variants', default='Ma,H1,Oa,Pg,Ua')
    args = ap.parse_args()
    items = [json.loads(l) for l in Path(args.items).read_text().splitlines() if l.strip()]
    if args.limit: items = items[:args.limit]
    import torch
    from transformers import AutoModelForCausalLM
    tok = C.load_tokenizer(); LABEL_IDS = dict(zip(('A', 'B', 'C', 'D'), range(32, 36))); rs_ids = H3.rs_prefix_ids(tok, LABEL_IDS)
    model = AutoModelForCausalLM.from_pretrained(str(C.MODEL_SNAPSHOT), local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation='eager', device_map={'': 0}, low_cpu_mem_usage=True)
    model.eval(); model.config.use_cache = False
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for vname in args.variants.split(','):
        ph, an, ids_kept, phrases = [], [], [], []
        for it in items:
            V = it['variants']
            if vname not in V: continue
            # Mdup has two insertions (mid paraphrase + the end-slot cue); the probe reads the END-slot cue = the Ma sentence
            sent = inserted_sentence(V['C'], V['Ma'] if vname == 'Mdup' else V[vname]); a, b = phrase_span_in(sent)
            prompt_ids, _p, text = C.prompt_ids_and_option_positions(tok, V[vname], it['options']); ids = prompt_ids + rs_ids
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True); offs = enc['offset_mapping']
            s0 = text.rindex(sent); pos = C.overlapping(offs, (s0 + a, s0 + b))
            if not pos: continue
            with torch.no_grad():
                o = model(torch.tensor([ids], device='cuda:0'), output_hidden_states=True, use_cache=False)
            hs = torch.stack(o.hidden_states, 0)[:, 0]                                      # [33, seq, 4096]
            ph.append(hs[:, pos, :].mean(1).float().cpu().numpy().astype(np.float16))
            an.append(hs[:, -1, :].float().cpu().numpy().astype(np.float16))
            ids_kept.append(it['item_id']); phrases.append(sent[a:b])
        np.savez_compressed(out / f'{vname}.npz', phrase=np.stack(ph), answer=np.stack(an), item_ids=np.array(ids_kept), phrases=np.array(phrases))
        print(f'{vname}: {len(ids_kept)} items  {time.time() - t0:.0f}s  e.g. phrase="{phrases[0]}"', flush=True)
    (out / 'RUN_INFO.json').write_text(json.dumps(dict(script='extract_hidden_v5.py', items=str(args.items), n=len(items), seconds=round(time.time() - t0)), indent=1))

if __name__ == '__main__':
    main()
