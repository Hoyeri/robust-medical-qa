# -*- coding: utf-8 -*-
"""Pre-registered analysis of the transmission audit (CM_PAPER_POSITION_20260920_KO.md §12.13). Written before the rows were read.

A. role decodability, M vs H1, at each stage (h, v, av, oav) and layer: by-item 5-fold, concept (HPO) holdout, attribution-template
   holdout (coworker <-> classmate/child), shuffled-label control. Standardize -> PCA(k=24, train only) -> L2 logistic.
   Verdict per stage in layers 8-16: retained (holdout acc >= 0.85), lost (<= 0.60), partial otherwise.
B. decision alignment A_e^(l) (first-order dq/dα of amplifying e's transmitted contribution at layer l on all receivers):
   paired within item, sum over layers 8-16 (primary) and per layer (secondary): A_M − A_H1 (53), A_M − A_U (118), A_Gp − A_U (70),
   A_M − A_rand (118), A_O − A_U (118). Also the answer-slot-only version A_ans. Implementation check from validation.json.
Usage: python3 -B analyze_transmission.py <results_dir> <items_jsonl> [--k 24]"""
import sys, json, re, random, statistics as st, numpy as np
from pathlib import Path
R = Path(sys.argv[1]); ITEMS = Path(sys.argv[2]); K = int(sys.argv[sys.argv.index('--k') + 1]) if '--k' in sys.argv else 24
items = {json.loads(l)['item_id']: json.loads(l) for l in ITEMS.read_text().splitlines() if l.strip()}
AL = json.loads((R / 'alignment.json').read_text()); info = json.loads((R / 'RUN_INFO.json').read_text())
print(f"{R.name}: n {info['n']} variants {info['variants']} {info['seconds']}s")
# ---- implementation check --------------------------------------------------------------------------------------------
val = json.loads((R / 'validation.json').read_text()) if (R / 'validation.json').exists() else []
if val:
    print('\n== implementation check: actual Δq at α=0.1 vs first-order α A_all (M sentence, layers 8/12/16) ==')
    for v in val: print(f"  {v['item_id']:24s} L{v['layer']:2d}  actual {v['dq_actual']:+.4f}  first-order {v['dq_first_order']:+.4f}")
    xs, ys = [v['dq_actual'] for v in val], [v['dq_first_order'] for v in val]
    if len(xs) > 2: print(f"  Pearson r = {np.corrcoef(xs, ys)[0, 1]:.3f}, median |actual − first-order| = {st.median(abs(a - b) for a, b in zip(xs, ys)):.4f}")
# ---- A. decodability ---------------------------------------------------------------------------------------------------
def load(v):
    z = np.load(R / f'{v}.npz'); return {k: z[k].astype(np.float32) for k in ('h', 'v', 'av', 'oav')} | dict(ids=list(z['item_ids']))
A, B = load('Ma'), load('H1'); ids = [i for i in A['ids'] if i in set(B['ids'])]; ia = [A['ids'].index(i) for i in ids]; ib = [B['ids'].index(i) for i in ids]
def h1_sentence(it):
    c, t = re.sub(r'\s+', ' ', it['variants']['C']).strip(), re.sub(r'\s+', ' ', it['variants']['H1']).strip(); i = 0
    while i < len(c) and c[i] == t[i]: i += 1
    j = 0
    while j < len(c) - i and c[len(c) - 1 - j] == t[len(t) - 1 - j]: j += 1
    return t[i:len(t) - j].strip()
concept = np.array([items[i]['meta']['cue']['hpo'] for i in ids]); family = np.array(['coworker' if 'coworker' in h1_sentence(items[i]) else 'classmate_child' for i in ids])
print(f"\n== A. M vs H1 decodability: pairs {len(ids)}, concepts {len(set(concept))}, families {dict(zip(*np.unique(family, return_counts=True)))} (PCA k={K}) ==")
def logreg(X, y, l2=1.0, iters=25):
    n, d = X.shape; Xb = np.hstack([X, np.ones((n, 1))]); w = np.zeros(d + 1); Rg = l2 * np.eye(d + 1); Rg[-1, -1] = 0
    for _ in range(iters):
        z = np.clip(Xb @ w, -30, 30); p = 1 / (1 + np.exp(-z)); w -= np.linalg.solve((Xb * (p * (1 - p))[:, None]).T @ Xb + Rg, Xb.T @ (p - y) + Rg @ w)
    return w
def fit_eval(XA_tr, XB_tr, XA_te, XB_te):
    Xtr = np.vstack([XA_tr, XB_tr]); ytr = np.r_[np.ones(len(XA_tr)), np.zeros(len(XB_tr))]; Xte = np.vstack([XA_te, XB_te]); yte = np.r_[np.ones(len(XA_te)), np.zeros(len(XB_te))]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = np.nan_to_num(np.clip((Xtr - mu) / sd, -20, 20)); Xte = np.nan_to_num(np.clip((Xte - mu) / sd, -20, 20))
    _, _, Vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False); P = Vt[:K].T; w = logreg(Xtr @ P, ytr)
    return ((np.hstack([Xte @ P, np.ones((len(Xte), 1))]) @ w > 0) == yte).mean()
def grouped_cv(XA, XB, groups, folds=5, seed=0, shuffle=False):
    rng = np.random.default_rng(seed); ug = list(dict.fromkeys(groups)); rng.shuffle(ug); fold_of = {g: k % folds for k, g in enumerate(ug)}; f = np.array([fold_of[g] for g in groups]); accs = []
    for k in range(folds):
        tr, te = f != k, f == k
        if shuffle:   # shuffled-label control: randomly swap M/H1 within each training pair
            sw = rng.random(tr.sum()) < 0.5; XA_tr, XB_tr = XA[tr].copy(), XB[tr].copy(); XA_tr[sw], XB_tr[sw] = XB[tr][sw], XA[tr][sw]
        else: XA_tr, XB_tr = XA[tr], XB[tr]
        accs.append(fit_eval(XA_tr, XB_tr, XA[te], XB[te]))
    return float(np.mean(accs))
def holdout(XA, XB, fam, train_fam):
    tr, te = fam == train_fam, fam != train_fam; return fit_eval(XA[tr], XB[tr], XA[te], XB[te]), int(tr.sum()), int(te.sum())
stage_layers = {'h': list(range(33)), 'v': list(range(32)), 'av': list(range(32)), 'oav': list(range(32))}
verdict = {}
for stage in ('h', 'v', 'av', 'oav'):
    XA_all, XB_all = A[stage][ia], B[stage][ib]
    print(f"\n-- stage {stage:3s}  {'layer':>5} {'by-item':>8} {'by-concept':>11} {'cow->cls':>9} {'cls->cow':>9} {'shuffled':>9}")
    hold = []
    for L in (0, 4, 8, 10, 12, 14, 16, 20, 24, 31):
        if L >= XA_all.shape[1]: continue
        XA, XB = XA_all[:, L], XB_all[:, L]
        bi = np.mean([grouped_cv(XA, XB, np.arange(len(ids)), seed=s) for s in (0, 1, 2)]); bc = np.mean([grouped_cv(XA, XB, concept, seed=s) for s in (0, 1, 2)])
        a1, _, _ = holdout(XA, XB, family, 'coworker'); a2, _, _ = holdout(XA, XB, family, 'classmate_child'); sh = np.mean([grouped_cv(XA, XB, np.arange(len(ids)), seed=s, shuffle=True) for s in (0, 1, 2)])
        print(f"   {'':10s} {L:>5} {bi:>8.2f} {bc:>11.2f} {a1:>9.2f} {a2:>9.2f} {sh:>9.2f}")
        if 8 <= L <= 16: hold.append(min(bc, a1, a2))
    m = float(np.mean(hold)); verdict[stage] = 'retained' if m >= 0.85 else ('lost' if m <= 0.60 else 'partial')
    print(f"   -> layers 8-16 mean of min(holdout accs) = {m:.2f} -> {verdict[stage]}")
print('\n== A verdict per stage (8-16):', verdict)
# ---- B. decision alignment ---------------------------------------------------------------------------------------------
def ci(xs, Bn=5000, seed=20260921):
    rnd = random.Random(seed); n = len(xs); bs = sorted(st.median(rnd.choices(xs, k=n)) for _ in range(Bn)); return st.median(xs), bs[int(.025 * Bn)], bs[int(.975 * Bn)]
def fmt(xs): m, lo, hi = ci(xs); return f'{m:+.4f} [{lo:+.4f}, {hi:+.4f}] (n={len(xs)}, >0 {sum(x > 0 for x in xs)})'
def Avec(i, variant, source, key): r = AL.get(i, {}).get(variant); return None if not r or source not in r['sources'] else r['sources'][source][key]
LAY = list(range(8, 17))
print('\n== B. decision alignment A_e = dq/dα of amplifying e\'s transmitted contribution (layers 8-16 summed; A_all = all receivers, A_ans = answer slot) ==')
for key in ('A_all', 'A_ans'):
    print(f'  [{key}]')
    for src in ('sent', 'phr'):
        print(f'   source span = {src}')
        for lbl, va, sa, vb, sb in (('A_M − A_H1', 'Ma', src, 'H1', src), ('A_M − A_U', 'Ma', src, 'Ua', src), ('A_Gp − A_U', 'Pg', src, 'Ua', src), ('A_O − A_U', 'Oa', src, 'Ua', src), ('A_M − A_rand', 'Ma', src, 'Ma', 'rand'), ('A_H1 − A_U', 'H1', src, 'Ua', src)):
            xs = []
            for i in AL:
                a, b = Avec(i, va, sa, key), Avec(i, vb, sb, key)
                if a is None or b is None: continue
                xs.append(sum(a[l] for l in LAY) - sum(b[l] for l in LAY))
            if len(xs) >= 4: print(f'     {lbl:14s} {fmt(xs)}')
    print('   raw A per source (8-16 sum, median):', ', '.join(f"{v}/{s}: {st.median(sum(Avec(i, v, s, key)[l] for l in LAY) for i in AL if Avec(i, v, s, key)):+.4f}" for v, s in (('Ma', 'sent'), ('H1', 'sent'), ('Ua', 'sent'), ('Oa', 'sent'), ('Pg', 'sent'), ('Ma', 'rand'))))
print('\n-- per-layer A_all (sentence span), median across items --')
print(f"  {'layer':>5} " + ' '.join(f'{v:>8s}' for v in ('M', 'H1', 'U', 'O', 'Gp', 'rand')) + '   M−H1(paired)  M−U(paired)')
for l in range(32):
    row = [st.median(Avec(i, v, s, 'A_all')[l] for i in AL if Avec(i, v, s, 'A_all')) for v, s in (('Ma', 'sent'), ('H1', 'sent'), ('Ua', 'sent'), ('Oa', 'sent'), ('Pg', 'sent'), ('Ma', 'rand'))]
    mh = st.median(Avec(i, 'Ma', 'sent', 'A_all')[l] - Avec(i, 'H1', 'sent', 'A_all')[l] for i in AL if Avec(i, 'H1', 'sent', 'A_all'))
    mu = st.median(Avec(i, 'Ma', 'sent', 'A_all')[l] - Avec(i, 'Ua', 'sent', 'A_all')[l] for i in AL if Avec(i, 'Ua', 'sent', 'A_all') and Avec(i, 'Ma', 'sent', 'A_all'))
    print(f"  {l:>5} " + ' '.join(f'{x:+8.4f}' for x in row) + f"   {mh:+.4f}      {mu:+.4f}" + ('   <- window' if 8 <= l <= 16 else ''))
