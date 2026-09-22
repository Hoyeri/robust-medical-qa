# -*- coding: utf-8 -*-
"""Step 1: is contextual evidence value decodable from hidden states? numpy-only (no sklearn on either machine).

Contrasts (pairs of variants, one sample per item per variant, folds split BY ITEM so both members of a pair fall in the
same fold):
  M vs H1   same finding phrase, different contextual role (patient vs coworker)   <- the informative contrast
  M vs O    T-supporting vs mundane finding (content differs: lexical sanity check, expected easy)
  Gp vs O   G-supporting vs mundane (content differs)
  M vs U    clinical vs non-clinical inserted sentence (content differs)
Positions: 'phrase' = mean over finding-phrase tokens (for M vs H1 the strings are identical, so layer 0 must be ~chance);
'answer' = answer-slot state. Classifier: standardize -> PCA(k) fit on the training fold -> L2 logistic regression (Newton
steps). 5-fold CV by item, repeated with 3 seeds; report accuracy per layer with min/max over repeats, and the chance
reference from label-shuffled runs.
Usage: python3 -B probe_contextual_role.py <hidden_dir> [--k 24]"""
import sys, json, numpy as np
from pathlib import Path
H = Path(sys.argv[1]); K = int(sys.argv[sys.argv.index('--k') + 1]) if '--k' in sys.argv else 24
def load(v):
    z = np.load(H / f'{v}.npz'); return dict(phrase=z['phrase'].astype(np.float32), answer=z['answer'].astype(np.float32), ids=list(z['item_ids']))
D = {v: load(v) for v in ('Ma', 'H1', 'Oa', 'Pg', 'Ua') if (H / f'{v}.npz').exists()}

def logreg(X, y, l2=1.0, iters=25):
    n, d = X.shape; Xb = np.hstack([X, np.ones((n, 1))]); w = np.zeros(d + 1); R = l2 * np.eye(d + 1); R[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w)); g = Xb.T @ (p - y) + R @ w
        Hm = (Xb * (p * (1 - p))[:, None]).T @ Xb + R
        w -= np.linalg.solve(Hm, g)
    return w
def predict(w, X): return (np.hstack([X, np.ones((len(X), 1))]) @ w) > 0

def cv_accuracy(XA, XB, seeds=(0, 1, 2), folds=5, shuffle_labels=False):
    n = len(XA); accs = []
    for seed in seeds:
        rng = np.random.default_rng(seed); order = rng.permutation(n); fold_of = np.empty(n, int); fold_of[order] = np.arange(n) % folds
        correct = 0
        for f in range(folds):
            tr, te = fold_of != f, fold_of == f
            Xtr = np.vstack([XA[tr], XB[tr]]); ytr = np.r_[np.ones(tr.sum()), np.zeros(tr.sum())]
            Xte = np.vstack([XA[te], XB[te]]); yte = np.r_[np.ones(te.sum()), np.zeros(te.sum())]
            if shuffle_labels: ytr = rng.permutation(ytr)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
            U, S, Vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False); P = Vt[:K].T
            w = logreg(Xtr @ P, ytr); correct += (predict(w, Xte @ P) == yte).sum()
        accs.append(correct / (2 * n))
    return float(np.mean(accs)), float(np.min(accs)), float(np.max(accs))

def contrast(a, b, pos):
    ids = [i for i in D[a]['ids'] if i in set(D[b]['ids'])]
    ia = [D[a]['ids'].index(i) for i in ids]; ib = [D[b]['ids'].index(i) for i in ids]
    XA, XB = D[a][pos][ia], D[b][pos][ib]                    # [n, 33, 4096]
    print(f'\n== {a} vs {b} @ {pos}   n={len(ids)} items per class, PCA k={K}, 5-fold by item x 3 seeds')
    print(f"  {'layer':>5} {'acc':>6} {'min':>6} {'max':>6}   {'shuffled':>8}")
    best = (0, 0)
    for L in range(XA.shape[1]):
        m, lo, hi = cv_accuracy(XA[:, L], XB[:, L]); s, _, _ = cv_accuracy(XA[:, L], XB[:, L], seeds=(7,), shuffle_labels=True)
        flag = ' <-' if m > best[1] else ''
        if m > best[1]: best = (L, m)
        if L in (0, 4, 8, 12, 16, 20, 24, 28, 32) or flag: print(f'  {L:>5} {m:>6.2f} {lo:>6.2f} {hi:>6.2f}   {s:>8.2f}{flag}')
    print(f'  best layer {best[0]} acc {best[1]:.2f}')
    return best
res = {}
for a, b in (('Ma', 'H1'), ('Ma', 'Oa'), ('Pg', 'Oa'), ('Ma', 'Ua')):
    if a in D and b in D:
        for pos in ('phrase', 'answer'): res[f'{a}-{b}@{pos}'] = contrast(a, b, pos)
print('\nSUMMARY (best layer, accuracy):', json.dumps({k: [int(v[0]), round(v[1], 3)] for k, v in res.items()}))
