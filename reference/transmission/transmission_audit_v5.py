# -*- coding: utf-8 -*-
"""Transmission audit (pre-registered: CM_PAPER_POSITION_20260920_KO.md §12.13).

For every item and variant (Ma, H1, Ua, Pg, Oa), ONE forward (eager attention, hidden states) + ONE backward of
q = z_T − z_G at the rs answer slot (gradients flow through the input embeddings only; no parameter grads).
Captured per layer l (0..31) from the attention module: the pre-repeat value tensor [1, 8, seq, 128] and the post-softmax
weights [1, 32, seq, seq]; from o_proj: its output tensor with retained grad ∂q/∂attn_out [1, seq, 4096].

Saved per variant (<out>/<variant>.npz, float16), features from the FINDING-PHRASE tokens of the inserted sentence
(template words excluded, so Ma and H1 have identical strings):
    h    [n, 33, 4096]  residual input of layer l at the phrase tokens (mean)          (l = 0 embeddings .. 32 final)
    v    [n, 32, 1024]  value vectors W_V LN(h) at the phrase tokens (mean; 8 KV heads × 128)
    av   [n, 32, 4096]  answer-slot contribution c_T = Σ_{j∈phrase} a_Tj v_j, heads concatenated (head h ↔ kv h//4)
    oav  [n, 32, 4096]  W_O c_T  (the phrase tokens' direct contribution to the answer-slot residual at layer l)
Alignment (<out>/alignment.json): for source e = full inserted sentence ('sent') and phrase tokens ('phr'), per layer:
    A_all[l] = Σ_{t∈receivers} inner(∂q/∂attn_out_t^(l), W_O c_{t,e}^(l))     receivers = all positions after the source
    A_ans[l] = inner(∂q/∂attn_out_T^(l), W_O c_{T,e}^(l))                       answer slot only
For the Ma variant a 'rand' source (an original vignette sentence of matched token count) is added.
Per-head decomposition (A_all_head / A_ans_head, layers 8-16 only): A^(h) = Σ_t inner(∂q/∂attn_out_t W_O, c_t^(h)), exact since W_O c is linear in head blocks.
--validate N: for the first N items, actually add α W_O c_{t,M} (α ∈ {1, 2, 4}) at layers 8/12/16 to attn_out and compare Δq with α A_all.
Usage (Hamster): CUDA_VISIBLE_DEVICES=2 python deps/transmission_audit_v5.py items/cm_confirmatory_v5.jsonl --out results/transmission_v5 [--validate 5] [--limit N]
"""
import argparse, hashlib, json, os, re, sys, time
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
from ariel_gsweep_fresh import generation_queries
from ariel_beta_hooks import require

LETTERS = ('A', 'B', 'C', 'D'); NL = 32; NH = 32; NKV = 8; HD = 128; HEAD_L0, HEAD_L1 = 8, 16   # per-head alignment saved for the pre-registered window only
SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])'); _norm = lambda t: re.sub(r'\s+', ' ', str(t)).strip()
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
    m = TEMPLATE.match(sentence); start = m.end() if m else 0
    body = re.sub(r'\s+(is also noted|is also seen)$', '', sentence.rstrip('.').rstrip()); return start, len(body)
def random_control_sentence(tok, c_text, m_sentence):
    sents = [s.strip() for s in SENT.split(_norm(c_text)) if s.strip()]
    if len(sents) > 1: sents = sents[:-1]
    n_m = len(tok(m_sentence, add_special_tokens=False)['input_ids'])
    return min(sents, key=lambda s: (abs(len(tok(s, add_special_tokens=False)['input_ids']) - n_m), sents.index(s)))

class Capture:
    """patch eager attention to record value (pre-repeat) and weights per layer; hooks on o_proj retain grads"""
    def __init__(self, torch, model):
        self.torch, self.model = torch, model; self.value, self.weights, self.oproj = {}, {}, {}; self.add = {}   # add: layer -> (rows, tensor) for --validate
        from transformers.models.llama import modeling_llama as impl
        self.impl, self.original = impl, impl.eager_attention_forward
        require(not getattr(self.original, '_discovery_wrapper', False), 'Nested attention wrapper')
        cap = self
        def wrapped(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
            out = cap.original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            cap.value[module.layer_idx] = value.detach(); cap.weights[module.layer_idx] = out[1].detach(); return out
        wrapped._discovery_wrapper = True; self.wrapped = wrapped
        self.handles = []
        for l, layer in enumerate(model.model.layers):
            def hook(mod, inp, out, l=l):
                if l in cap.add:
                    rows, extra = cap.add[l]; out = out.clone(); out[0, rows, :] = out[0, rows, :] + extra.to(out.dtype)
                if out.requires_grad: out.retain_grad()
                cap.oproj[l] = out; return out
            self.handles.append(layer.self_attn.o_proj.register_forward_hook(hook))
        self.handles.append(model.model.embed_tokens.register_forward_hook(lambda m, i, o: o.requires_grad_(True) if cap.torch.is_grad_enabled() else None))
    def __enter__(self): self.impl.eager_attention_forward = self.wrapped; return self
    def __exit__(self, *a): self.impl.eager_attention_forward = self.original
    def clear(self): self.value.clear(); self.weights.clear(); self.oproj.clear()
    def close(self):
        for h in self.handles: h.remove()

def contributions(torch, cap, WO, l, src, Q):
    """c_t (heads concatenated) and W_O c_t for receivers Q from source positions src, at layer l  -> [|Q|, 4096] each (float32)"""
    a = cap.weights[l][0][:, Q][:, :, src].float()                                   # [32, |Q|, |src|]
    v = cap.value[l][0][:, src, :].float().repeat_interleave(NH // NKV, dim=0)      # [32, |src|, 128]  head h <- kv h // 4
    c = torch.einsum('hqs,hsd->qhd', a, v).reshape(len(Q), NH * HD)                  # [|Q|, 4096]
    return c, c @ WO[l].T

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('items'); ap.add_argument('--out', required=True); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--validate', type=int, default=0); ap.add_argument('--variants', default='Ma,H1,Ua,Pg,Oa')
    args = ap.parse_args()
    items = [json.loads(l) for l in Path(args.items).read_text().splitlines() if l.strip()]
    if args.limit: items = items[:args.limit]
    import torch
    from transformers import AutoModelForCausalLM
    tok = C.load_tokenizer(); LABEL_IDS = dict(zip(LETTERS, range(32, 36))); rs_ids = H3.rs_prefix_ids(tok, LABEL_IDS)
    model = AutoModelForCausalLM.from_pretrained(str(C.MODEL_SNAPSHOT), local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation='eager', device_map={'': 0}, low_cpu_mem_usage=True)
    model.eval(); model.config.use_cache = False
    for p in model.parameters(): p.requires_grad_(False)
    WO = [model.model.layers[l].self_attn.o_proj.weight.detach().float() for l in range(NL)]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time(); align = {}; validation = []
    cap = Capture(torch, model)
    feats = {v: dict(h=[], v=[], av=[], oav=[], ids=[], phrases=[]) for v in args.variants.split(',')}
    with cap:
        for n, it in enumerate(items, 1):
            V = it['variants']; G, T = it['gold_G'], it['target_T']; align[it['item_id']] = {}
            for vname in args.variants.split(','):
                if vname not in V: continue
                sent = inserted_sentence(V['C'], V['Ma'] if vname == 'Mdup' else V[vname]); a, b = phrase_span_in(sent)   # Mdup has two insertions; audit the end-slot Ma sentence
                prompt_ids, _p, text = C.prompt_ids_and_option_positions(tok, V[vname], it['options']); ids = prompt_ids + rs_ids; N = len(ids)
                enc = tok(text, add_special_tokens=False, return_offsets_mapping=True); offs = enc['offset_mapping']; s0 = text.rindex(sent)
                phr = C.overlapping(offs, (s0 + a, s0 + b)); full = H3.span_positions(tok, text, sent)
                if not phr or not full or 0 in full: continue
                sources = {'sent': full, 'phr': phr}
                if vname == 'Ma':
                    rs = random_control_sentence(tok, V['C'], sent)
                    try: sources['rand'] = H3.span_positions(tok, text, rs)
                    except Exception: pass
                cap.clear(); cap.add.clear()
                with torch.enable_grad():
                    o = model(torch.tensor([ids], device='cuda:0'), output_hidden_states=True, use_cache=False)
                    lg = o.logits[0, -1].float(); q = lg[LABEL_IDS[T]] - lg[LABEL_IDS[G]]; q.backward()
                hs = [h.detach() for h in o.hidden_states]; lab = {k: float(lg[j].detach()) for k, j in LABEL_IDS.items()}
                grads = {l: cap.oproj[l].grad[0].float() for l in range(NL)}                     # [seq, 4096]
                rec = dict(q=float(q), argmax=max(lab, key=lab.get), rs_label_logits=lab, n_tokens=N, sources={})
                for sname, src in sources.items():
                    Q = generation_queries(N, src); A_all, A_ans, Ah_all, Ah_ans = [], [], {}, {}
                    for l in range(NL):
                        c, ct = contributions(torch, cap, WO, l, src, Q); g = grads[l][Q]
                        A_all.append(float((g * ct).sum())); A_ans.append(float((g[-1] * ct[-1]).sum()))
                        if HEAD_L0 <= l <= HEAD_L1:      # per-head decomposition: c̃ = Σ_h W_O[:, h-block] c^(h)  (exact, linear)
                            gW = g @ WO[l]                                                   # [|Q|, 4096] = (∂q/∂attn_out) W_O  -> dot with c per head block
                            ph = (gW * c).reshape(len(Q), NH, HD).sum(-1)                    # [|Q|, 32]
                            Ah_all[str(l)] = ph.sum(0).tolist(); Ah_ans[str(l)] = ph[-1].tolist()
                    rec['sources'][sname] = dict(n_span_tokens=len(src), A_all=A_all, A_ans=A_ans, A_all_head=Ah_all, A_ans_head=Ah_ans)
                align[it['item_id']][vname] = rec
                # phrase-token features
                F = feats[vname]; F['ids'].append(it['item_id']); F['phrases'].append(sent[a:b])
                F['h'].append(torch.stack([hs[l][0, phr, :].float().mean(0) for l in range(NL + 1)]).cpu().numpy().astype(np.float16))
                F['v'].append(torch.stack([cap.value[l][0][:, phr, :].float().mean(1).reshape(-1) for l in range(NL)]).cpu().numpy().astype(np.float16))
                avs, oavs = [], []
                for l in range(NL):
                    c, ct = contributions(torch, cap, WO, l, phr, [N - 1]); avs.append(c[0]); oavs.append(ct[0])
                F['av'].append(torch.stack(avs).cpu().numpy().astype(np.float16)); F['oav'].append(torch.stack(oavs).cpu().numpy().astype(np.float16))
                # implementation check of the first-order alignment: actually add α W_O c_{t,M} at one layer
                if args.validate and n <= args.validate and vname == 'Ma':
                    for l in (8, 12, 16):
                        Q = generation_queries(N, full); _, ct = contributions(torch, cap, WO, l, full, Q)
                        for alpha in (1.0, 2.0, 4.0):        # bf16 logits are quantised to 0.125, so α must be large enough for Δq to exceed the quantum
                            cap.add = {l: (Q, alpha * ct)}
                            with torch.no_grad():
                                lg2 = model(torch.tensor([ids], device='cuda:0'), use_cache=False).logits[0, -1].float()
                            cap.add = {}
                            dq = float(lg2[LABEL_IDS[T]] - lg2[LABEL_IDS[G]]) - float(q)
                            validation.append(dict(item_id=it['item_id'], layer=l, alpha=alpha, dq_actual=dq, dq_first_order=alpha * rec['sources']['sent']['A_all'][l]))
                del o, hs, grads; torch.cuda.empty_cache()
            if n % 10 == 0 or n == len(items): print(f"{n}/{len(items)} {it['item_id']} {time.time() - t0:.0f}s", flush=True)
    cap.close()
    for vname, F in feats.items():
        if F['ids']: np.savez_compressed(out / f'{vname}.npz', h=np.stack(F['h']), v=np.stack(F['v']), av=np.stack(F['av']), oav=np.stack(F['oav']), item_ids=np.array(F['ids']), phrases=np.array(F['phrases']))
    (out / 'alignment.json').write_text(json.dumps(align)); (out / 'validation.json').write_text(json.dumps(validation, indent=1))
    info = dict(script='transmission_audit_v5.py', script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), items=str(args.items), items_sha256=hashlib.sha256(Path(args.items).read_bytes()).hexdigest(),
                n=len(items), variants=args.variants, model_snapshot=str(C.MODEL_SNAPSHOT), device=torch.cuda.get_device_name(0), rs_prefix=H3.RS_PREFIX, seconds=round(time.time() - t0), started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    (out / 'RUN_INFO.json').write_text(json.dumps(info, indent=1)); print(json.dumps(info), flush=True)
    if validation: print('validation (α=0.1 actual Δq vs first-order α A_all):', ' '.join(f"L{v['layer']}:{v['dq_actual']:+.3f}/{v['dq_first_order']:+.3f}" for v in validation[:15]), flush=True)

if __name__ == '__main__':
    main()
