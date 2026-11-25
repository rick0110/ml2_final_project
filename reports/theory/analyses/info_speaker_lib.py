"""functions extracted from info_speaker.py (no side effects)"""
import numpy as np
from scipy.linalg import eigh
from info_common import *
def scatter(X, lab):
    """returns speakers, counts, means, within scatter SSW, between scatter SSB"""
    u, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    M = np.zeros((len(u), X.shape[1])); np.add.at(M, inv, X); M /= cnt[:, None]
    R = X - M[inv]; SSW = R.T @ R
    D = M - X.mean(0); SSB = (D * cnt[:, None]).T @ D
    return u, inv, cnt, M, SSW, SSB

def gauss_estimates(X, lab, noise=None):
    """plug-in (T vs W) and variance-component (bias-corrected) Gaussian estimates.
    noise: optional (p,) posterior variance added to W (channel z = mu + sigma*eps)."""
    u, inv, cnt, M, SSW, SSB = scatter(X, lab)
    n, p = X.shape; S = len(u)
    W = SSW / (n - S); T = (SSW + SSB) / (n - 1)
    if noise is not None:
        W = W + np.diag(noise); T = T + np.diag(noise)
    ev = eigh(T, W, eigvals_only=True)                       # generalized eigenvalues of total vs within
    I_plug = 0.5 * np.log2(np.clip(ev, 1e-12, None)).sum()
    n0 = (n - (cnt**2).sum() / n) / (S - 1)
    MSB = SSB / (S - 1)
    Wm = SSW / (n - S)
    Sb = (MSB - Wm) / n0                                      # unbiased between-speaker covariance (can be indefinite)
    lam = eigh(Sb, W, eigvals_only=True)[::-1]                # between/within SNR per direction
    I_vc = 0.5 * np.log2(1 + np.clip(lam, 0, None)).sum()
    lam_raw = np.linalg.eigvalsh(Sb)[::-1]
    return dict(I_plug=float(I_plug), I_vc=float(I_vc), lam=lam, lam_raw=lam_raw, ev=ev[::-1], Sb=Sb, W=W, n0=float(n0), S=S, n=n)

def loo_lda(X, lab, noise_rng=None, sig2=None):
    """exact leave-one-out Gaussian classifier with shared within covariance; predictive covariance W(1+1/n_s).
    returns accuracy, cross-entropy (bits), top-1 preds. If noise_rng is given, test point is x+sigma*eps (channel z)."""
    u, inv, cnt, M, SSW, SSB = scatter(X, lab)
    n, p = X.shape; S = len(u)
    logp_true = np.zeros(n); correct = np.zeros(n, bool)
    for i in range(n):
        s = inv[i]; ns = cnt[s]
        d = X[i] - M[s]
        SSWi = SSW - (ns / (ns - 1.0)) * np.outer(d, d)
        Wi = SSWi / (n - 1 - S)
        if sig2 is not None: Wi = Wi + np.diag(sig2.mean(0))
        Mi = M.copy(); Mi[s] = (M[s] * ns - X[i]) / (ns - 1)
        ci = cnt.astype(float).copy(); ci[s] -= 1
        L = np.linalg.cholesky(Wi)
        x = X[i] if noise_rng is None else X[i] + np.sqrt(sig2[i]) * noise_rng.standard_normal(p)
        Y = np.linalg.solve(L, (x - Mi).T).T                  # (S,p)
        q = (Y**2).sum(1)
        f = 1 + 1 / ci
        ll = -0.5 * q / f - 0.5 * p * np.log(f) + np.log(ci / ci.sum())
        ll -= ll.max(); lp = ll - np.log(np.exp(ll).sum())
        logp_true[i] = lp[s]; correct[i] = lp.argmax() == s
    return float(correct.mean()), float(-logp_true.mean() / LOG2)

def fano_bits(H, acc, S):
    pe = min(max(1 - acc, 1e-12), 1 - 1e-12)
    hb = -(pe * np.log2(pe) + (1 - pe) * np.log2(1 - pe))
    return H - hb - pe * np.log2(max(S - 1, 1))

def logreg_cv(X, lab):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    u, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    k = int(min(5, cnt.min()))
    if k < 2: return None
    accs, ces = [], []
    skf = StratifiedKFold(k, shuffle=True, random_state=0)
    lp = np.zeros(len(X)); ok = np.zeros(len(X), bool)
    for a, b in skf.split(X, inv):
        sc = StandardScaler().fit(X[a]); clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[a]), inv[a])
        P = clf.predict_log_proba(sc.transform(X[b])); col = {c: j for j, c in enumerate(clf.classes_)}
        for r, i in enumerate(b):
            lp[i] = P[r, col[inv[i]]] if inv[i] in col else np.log(1e-6)
            ok[i] = clf.classes_[P[r].argmax()] == inv[i]
    return float(ok.mean()), float(-lp.mean() / LOG2)

def twonn(P):
    """Facco et al. 2017 TwoNN, MLE form d = n / sum log(r2/r1)"""
    from scipy.spatial.distance import cdist
    D = cdist(P, P); np.fill_diagonal(D, np.inf); D.sort(1)
    r = D[:, 1] / D[:, 0]; r = r[np.isfinite(r) & (r > 1)]
    return float(len(r) / np.log(r).sum()) if len(r) > 2 else float("nan")
def levina_bickel(P, k=5):
    from scipy.spatial.distance import cdist
    D = cdist(P, P); np.fill_diagonal(D, np.inf); D.sort(1)
    k = min(k, len(P) - 1)
    if k < 3: return float("nan")
    m = (k - 2) / np.log(D[:, k - 1:k] / D[:, :k - 1]).sum(1)   # MacKay-Ghahramani unbiased version, per point
    return float(1 / np.mean(1 / m))
def powerlaw(lam, kmax=None):
    lam = np.asarray(lam); pos = lam > 0
    k = int(np.argmin(pos)) if not pos.all() else len(lam)    # leading positive run
    if kmax: k = min(k, kmax)
    if k < 4: return float("nan"), k
    i = np.arange(1, k + 1)
    return float(-np.polyfit(np.log(i), np.log(lam[:k]), 1)[0]), k

