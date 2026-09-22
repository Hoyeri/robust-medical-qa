"""Frozen numerical functions used only for regression comparison."""

def load(v):
    z = np.load(H / f'{v}.npz'); return dict(phrase=z['phrase'].astype(np.float32), answer=z['answer'].astype(np.float32), ids=list(z['item_ids']))

def h1_sentence(it):
    c, t = re.sub(r'\s+', ' ', it['variants']['C']).strip(), re.sub(r'\s+', ' ', it['variants']['H1']).strip(); i = 0
    while i < len(c) and c[i] == t[i]: i += 1
    j = 0
    while j < len(c) - i and c[len(c) - 1 - j] == t[len(t) - 1 - j]: j += 1
    return t[i:len(t) - j].strip()

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
