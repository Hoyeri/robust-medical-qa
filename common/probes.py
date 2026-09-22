"""Shared probing operations, with explicit historical numerical profiles."""
from dataclasses import dataclass
import numpy as np
from .io import require


@dataclass(frozen=True)
class ProbeConfig:
    components: int = 24
    l2: float = 1.0
    iterations: int = 25
    epsilon: float = 1e-6
    standard_clip: object = None
    logit_clip: object = None
    nan_to_num: bool = False
    aggregation: str = 'pooled'
    folds: int = 5
    seeds: tuple = (0, 1, 2)

    def __post_init__(self):
        require(self.components > 0 and self.iterations > 0 and self.folds >= 2, 'Invalid probe dimensions')
        require(self.l2 > 0 and self.epsilon > 0 and len(self.seeds) > 0, 'Invalid probe settings')
        require(self.aggregation in ('pooled', 'fold_mean'), 'Unknown aggregation')


def finite(*values):
    for value in values:
        if not np.isfinite(value).all():
            raise FloatingPointError('Non-finite probe intermediate')


def fit_logistic(X, y, config):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    R = config.l2 * np.eye(d + 1)
    R[-1, -1] = 0
    for _ in range(config.iterations):
        z = Xb @ w
        if config.logit_clip is not None:
            z = np.clip(z, -config.logit_clip, config.logit_clip)
        p = 1 / (1 + np.exp(-z))
        g = Xb.T @ (p - y) + R @ w
        hessian = (Xb * (p * (1 - p))[:, None]).T @ Xb + R
        finite(p, g, hessian)
        w -= np.linalg.solve(hessian, g)
        finite(w)
    return w


def fit_predict(Xtr, ytr, Xte, config):
    require(len(Xtr) > 1 and len(Xte) > 0, 'Empty training/evaluation split')
    mu, sd = Xtr.mean(0), Xtr.std(0) + config.epsilon
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if config.standard_clip is not None:
        Xtr = np.clip(Xtr, -config.standard_clip, config.standard_clip)
        Xte = np.clip(Xte, -config.standard_clip, config.standard_clip)
    if config.nan_to_num:
        Xtr, Xte = np.nan_to_num(Xtr), np.nan_to_num(Xte)
    finite(Xtr, Xte)
    _, _, Vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False)
    P = Vt[:config.components].T
    train, test = Xtr @ P, Xte @ P
    finite(P, train, test)
    w = fit_logistic(train, ytr, config)
    scores = np.hstack([test, np.ones((len(test), 1))]) @ w
    finite(scores)
    return scores > 0


def evaluate_split(XA, XB, train, test, config, rng=None, shuffle=None):
    a, b = XA[train].copy(), XB[train].copy()
    if shuffle == 'pair_swap':
        swap = rng.random(len(a)) < 0.5
        a[swap], b[swap] = b[swap], a[swap].copy()
    Xtr = np.vstack([a, b])
    ytr = np.r_[np.ones(len(a)), np.zeros(len(b))]
    if shuffle == 'labels':
        ytr = rng.permutation(ytr)
    yte = np.r_[np.ones(test.sum()), np.zeros(test.sum())]
    predictions = fit_predict(Xtr, ytr, np.vstack([XA[test], XB[test]]), config)
    return {'correct': int((predictions == yte).sum()), 'n': len(yte)}


def cross_validate(XA, XB, config, groups=None, shuffle=None):
    require(XA.shape == XB.shape and XA.ndim == 2, 'Expected aligned paired matrices')
    require(shuffle in (None, 'labels', 'pair_swap'), 'Unknown shuffle control')
    n = len(XA)
    groups = list(range(n)) if groups is None else list(groups)
    require(len(groups) == n, 'Group count mismatch')
    unique = list(dict.fromkeys(groups))
    require(len(unique) >= config.folds, 'Fewer groups than folds')
    repeats = []
    for seed in config.seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(unique))
        assignments = {unique[g]: i % config.folds for i, g in enumerate(order)}
        fold_of = np.array([assignments[g] for g in groups])
        folds = [evaluate_split(XA, XB, fold_of != f, fold_of == f, config, rng, shuffle)
                 for f in range(config.folds)]
        correct, total = sum(r['correct'] for r in folds), sum(r['n'] for r in folds)
        accuracy = correct / total if config.aggregation == 'pooled' else float(np.mean([r['correct'] / r['n'] for r in folds]))
        repeats.append({'seed': seed, 'correct': correct, 'n': total, 'accuracy': accuracy, 'folds': folds})
    return {'independent_pairs': n, 'mean_accuracy': float(np.mean([r['accuracy'] for r in repeats])),
            'aggregation': config.aggregation, 'repeats': repeats}


def holdout(XA, XB, families, train_family, config):
    train = np.array(families) == train_family
    require(train.any() and (~train).any(), 'Holdout requires training and evaluation families')
    result = evaluate_split(XA, XB, train, ~train, config)
    return {**result, 'accuracy': result['correct'] / result['n'], 'train_pairs': int(train.sum())}


def load_paired(directory, variants, stages):
    from pathlib import Path
    require(len(variants) == 2 and variants[0] != variants[1], 'Pairwise analysis requires exactly two distinct variants')
    data = []
    for variant in variants:
        with np.load(Path(directory) / f'{variant}.npz', allow_pickle=False) as z:
            ids = list(z['item_ids'])
            require(len(ids) == len(set(ids)), 'Duplicate item IDs')
            arrays = {stage: z[stage].astype(np.float32) for stage in stages}
            require(all(len(v) == len(ids) for v in arrays.values()), 'Feature/ID count mismatch')
            data.append((ids, arrays))
    ids = [i for i in data[0][0] if i in set(data[1][0])]
    require(bool(ids), 'No paired items')
    aligned = []
    for source_ids, arrays in data:
        index = {key: i for i, key in enumerate(source_ids)}
        aligned.append({stage: value[[index[i] for i in ids]] for stage, value in arrays.items()})
    return ids, aligned[0], aligned[1]
