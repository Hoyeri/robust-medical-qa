# -*- coding: utf-8 -*-
"""Is the M-vs-H1 decodability a template fingerprint? Three harder splits on the same hidden states:
  by-item (reference)        both members of an item in the same fold
  by-concept                 folds grouped by the cue's HPO id (same finding never in train and test)
  by-attribution-template    train on one attribution wording family, test on the other
                             (coworker-type: "Her/His/The patient's coworker" vs classmate/child-type:
                              "A classmate of ...", "Another child/baby at the clinic")
If the probe survives the template holdout, the phrase-token representation carries an abstract "not the patient" role,
not the word 'coworker'. Usage: python3 -B probe_robustness.py <hidden_dir> <items_jsonl> [--k 24]"""
import sys, json, re, numpy as np
from pathlib import Path
H = Path(sys.argv[1]); ITEMS = Path(sys.argv[2]); K = int(sys.argv[sys.argv.index('--k') + 1]) if '--k' in sys.argv else 24
items = {json.loads(l)['item_id']: json.loads(l) for l in ITEMS.read_text().splitlines() if l.strip()}
def load(v):
    z = np.load(H / f'{v}.npz'); return dict(phrase=z['phrase'].astype(np.float32), answer=z['answer'].astype(np.float32), ids=list(z['item_ids']))
A, B = load('Ma'), load('H1')
ids = [i for i in A['ids'] if i in set(B['ids'])]
ia = [A['ids'].index(i) for i in ids]; ib = [B['ids'].index(i) for i in ids]
def h1_sentence(it):
    c, t = re.sub(r'\s+', ' ', it['variants']['C']).strip(), re.sub(r'\s+', ' ', it['variants']['H1']).strip(); i = 0
    while i < len(c) and c[i] == t[i]: i += 1
    j = 0
    while j < len(c) - i and c[len(c) - 1 - j] == t[len(t) - 1 - j]: j += 1
    return t[i:len(t) - j].strip()
concept = np.array([items[i]['meta']['cue']['hpo'] for i in ids])
family = np.array(['coworker' if 'coworker' in h1_sentence(items[i]) else 'classmate_child' for i in ids])
print(f'pairs {len(ids)} | distinct cue concepts {len(set(concept))} | attribution families {dict(zip(*np.unique(family, return_counts=True)))}')

def logreg(X, y, l2=1.0, iters=25):
    n, d = X.shape; Xb = np.hstack([X, np.ones((n, 1))]); w = np.zeros(d + 1); R = l2 * np.eye(d + 1); R[-1, -1] = 0
    for _ in range(iters):
        z = np.clip(Xb @ w, -30, 30); p = 1 / (1 + np.exp(-z)); g = Xb.T @ (p - y) + R @ w
        w -= np.linalg.solve((Xb * (p * (1 - p))[:, None]).T @ Xb + R, g)
    return w
def fit_eval(XA_tr, XB_tr, XA_te, XB_te):
    Xtr = np.vstack([XA_tr, XB_tr]); ytr = np.r_[np.ones(len(XA_tr)), np.zeros(len(XB_tr))]
    Xte = np.vstack([XA_te, XB_te]); yte = np.r_[np.ones(len(XA_te)), np.zeros(len(XB_te))]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = np.clip((Xtr - mu) / sd, -20, 20); Xte = np.clip((Xte - mu) / sd, -20, 20)
    _, _, Vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False); P = Vt[:K].T
    w = logreg(Xtr @ P, ytr); return ((np.hstack([Xte @ P, np.ones((len(Xte), 1))]) @ w > 0) == yte).mean()

def grouped_cv(XA, XB, groups, folds=5, seed=0):
    rng = np.random.default_rng(seed); ug = list(dict.fromkeys(groups)); rng.shuffle(ug)
    fold_of_group = {g: k % folds for k, g in enumerate(ug)}; f = np.array([fold_of_group[g] for g in groups])
    acc, n = 0.0, 0
    for k in range(folds):
        tr, te = f != k, f == k
        if te.sum() == 0: continue
        acc += fit_eval(XA[tr], XB[tr], XA[te], XB[te]) * te.sum(); n += te.sum()
    return acc / n
def holdout(XA, XB, fam, train_fam):
    tr, te = fam == train_fam, fam != train_fam
    return fit_eval(XA[tr], XB[tr], XA[te], XB[te]), int(tr.sum()), int(te.sum())

for pos in ('phrase', 'answer'):
    XA, XB = A[pos][ia], B[pos][ib]
    print(f'\n== M vs H1 @ {pos}  (PCA k={K})')
    print(f"  {'layer':>5} {'by-item':>8} {'by-concept':>11} {'cow->cls':>9} {'cls->cow':>9}")
    for L in (0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32):
        bi = np.mean([grouped_cv(XA[:, L], XB[:, L], np.array(ids), seed=s) for s in (0, 1, 2)])
        bc = np.mean([grouped_cv(XA[:, L], XB[:, L], concept, seed=s) for s in (0, 1, 2)])
        a1, n1, m1 = holdout(XA[:, L], XB[:, L], family, 'coworker'); a2, n2, m2 = holdout(XA[:, L], XB[:, L], family, 'classmate_child')
        print(f'  {L:>5} {bi:>8.2f} {bc:>11.2f} {a1:>9.2f} {a2:>9.2f}' + (f'   (train n={n1}/{n2}, test n={m1}/{m2})' if L == 0 else ''))
