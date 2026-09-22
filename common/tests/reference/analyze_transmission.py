"""Frozen numerical functions used only for regression comparison."""

def load(v):
    z = np.load(R / f'{v}.npz'); return {k: z[k].astype(np.float32) for k in ('h', 'v', 'av', 'oav')} | dict(ids=list(z['item_ids']))

def h1_sentence(it):
    c, t = re.sub(r'\s+', ' ', it['variants']['C']).strip(), re.sub(r'\s+', ' ', it['variants']['H1']).strip(); i = 0
    while i < len(c) and c[i] == t[i]: i += 1
    j = 0
    while j < len(c) - i and c[len(c) - 1 - j] == t[len(t) - 1 - j]: j += 1
    return t[i:len(t) - j].strip()

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

def ci(xs, Bn=5000, seed=20260921):
    rnd = random.Random(seed); n = len(xs); bs = sorted(st.median(rnd.choices(xs, k=n)) for _ in range(Bn)); return st.median(xs), bs[int(.025 * Bn)], bs[int(.975 * Bn)]

def fmt(xs): m, lo, hi = ci(xs); return f'{m:+.4f} [{lo:+.4f}, {hi:+.4f}] (n={len(xs)}, >0 {sum(x > 0 for x in xs)})'

def Avec(i, variant, source, key): r = AL.get(i, {}).get(variant); return None if not r or source not in r['sources'] else r['sources'][source][key]
