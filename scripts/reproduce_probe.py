"""Portable M/H1 reanalysis, adapted from reference/role/probe_contextual_role.py.

Same fitting operations and split order; added CLI, per-seed counts and finite checks.
Only the indices documented in results/role_probe_exact.json are evaluated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(variable, '1')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def finite(*values):
    for value in values:
        if not np.isfinite(value).all():
            raise FloatingPointError('Non-finite intermediate in probe')


def fit(X, y):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    R = np.eye(d + 1)
    R[-1, -1] = 0
    for _ in range(25):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + R @ w
        Hm = (Xb * (p * (1 - p))[:, None]).T @ Xb + R
        finite(p, g, Hm)
        w -= np.linalg.solve(Hm, g)
        finite(w)
    return w


def counts(XA, XB):
    n = len(XA)
    result = []
    for seed in (0, 1, 2):
        order = np.random.default_rng(seed).permutation(n)
        fold_of = np.empty(n, int)
        fold_of[order] = np.arange(n) % 5
        correct = 0
        for fold in range(5):
            tr, te = fold_of != fold, fold_of == fold
            Xtr = np.vstack([XA[tr], XB[tr]])
            ytr = np.r_[np.ones(tr.sum()), np.zeros(tr.sum())]
            Xte = np.vstack([XA[te], XB[te]])
            yte = np.r_[np.ones(te.sum()), np.zeros(te.sum())]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            _, _, Vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False)
            P = Vt[:24].T
            train, test = Xtr @ P, Xte @ P
            finite(Xtr, Xte, P, train, test)
            w = fit(train, ytr)
            scores = np.hstack([test, np.ones((len(test), 1))]) @ w
            finite(scores)
            correct += int(((scores > 0) == yte).sum())
        result.append(correct)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hidden-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new filename to preserve prior results.')
    expected = json.loads((ROOT / 'results/role_probe_exact.json').read_text())
    arrays, hashes = {}, {}
    for name in ('Ma', 'H1'):
        path = args.hidden_dir / f'{name}.npz'
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as z:
            arrays[name] = {p: z[p].astype(np.float32) for p in ('phrase', 'answer')}
            arrays[name]['ids'] = list(z['item_ids'])
        assert len(set(arrays[name]['ids'])) == len(arrays[name]['ids'])
    a, b = arrays['Ma'], arrays['H1']
    ids = [i for i in a['ids'] if i in set(b['ids'])]
    assert len(ids) == 53, 'This script evaluates the frozen 53-pair experiment only.'
    ia, ib = [a['ids'].index(i) for i in ids], [b['ids'].index(i) for i in ids]
    rows = []
    for row in expected['rows']:
        pos, layer = row['position'], row['layer']
        observed = counts(a[pos][ia, layer], b[pos][ib, layer])
        same = observed == row['correct_per_seed']
        rows.append(dict(position=pos, layer=layer, correct_per_seed=observed,
                         denominator_per_seed=106, mean_accuracy=sum(observed)/318,
                         matches_frozen_counts=same))
        print(f'{pos} index {layer}: {observed}/106; match={same}', flush=True)
    report = dict(numpy_version=np.__version__, independent_pairs=53, seeds=[0, 1, 2],
                  folds=5, pca_k=24, source_sha256=hashes, rows=rows)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not all(r['matches_frozen_counts'] for r in rows):
        raise SystemExit('Counts differ; see output. Check input hashes and numerical environment.')
    print('PASS: all 12 selected probe rows match frozen per-seed counts; finite checks passed.')


if __name__ == '__main__':
    main()
