"""Tasks 2-3: speaker information estimators + speaker-manifold dimension.
usage: python3 -I info_speaker.py DATA OUT [nperm] [nboot]"""
import sys, os, json, warnings; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import eigh
from info_common import *
warnings.filterwarnings("ignore")
root, out = sys.argv[1], sys.argv[2]
NPERM = int(sys.argv[3]) if len(sys.argv) > 3 else 200
NBOOT = int(sys.argv[4]) if len(sys.argv) > 4 else 200

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

rng = np.random.default_rng(0)
res = {}
for run, N, K, seed, var in list_runs(root):
    d = load_latents(root, run); tr = d["train"]; spk = d["speaker"]
    lj = tr & (spk == "lj"); lib = tr & ~lj
    mu, s2 = d["mu"], np.exp(d["logvar"])
    r = dict(run=run, N=N, K=K, seed=seed, variant=var)
    # ---------- LibriTTS speakers only (speakers with >= 2 utterances)
    X, lab, sg = mu[lib], spk[lib], s2[lib]
    u, cnt = np.unique(lab, return_counts=True)
    keep = np.isin(lab, u[cnt >= 2]); X, lab, sg = X[keep], lab[keep], sg[keep]
    u, cnt = np.unique(lab, return_counts=True); S = len(u); n = len(X)
    H = entropy_bits(cnt)
    r.update(S=S, n=n, utt_per_spk=n / S, H=H, log2S=float(np.log2(S)))
    enough = (n - S) > 40                                     # W needs > 32 dof to be invertible
    r["enough"] = bool(enough)
    if enough:
        g = gauss_estimates(X, lab); gc = gauss_estimates(X, lab, noise=sg.mean(0))
        r.update(I_plug=g["I_plug"], I_vc=g["I_vc"], I_plug_chan=gc["I_plug"], I_vc_chan=gc["I_vc"], n0=g["n0"])
        # permutation null
        pp, pv = [], []
        for _ in range(NPERM):
            gp = gauss_estimates(X, rng.permutation(lab)); pp.append(gp["I_plug"]); pv.append(gp["I_vc"])
        r.update(I_plug_perm=float(np.mean(pp)), I_plug_perm_sd=float(np.std(pp)), I_vc_perm=float(np.mean(pv)), I_vc_perm_sd=float(np.std(pv)),
                 I_plug_p=float((np.array(pp) >= g["I_plug"]).mean()))
        acc, ce = loo_lda(X, lab); r.update(lda_acc=acc, lda_ce=ce, I_ce=H - ce, I_fano=fano_bits(H, acc, S))
        accz, cez = loo_lda(X, lab, noise_rng=np.random.default_rng(1), sig2=sg); r.update(ldaz_acc=accz, I_ce_chan=H - cez, I_fano_chan=fano_bits(H, accz, S))
        pa, pc = [], []
        for _ in range(max(5, NPERM // 20)):
            a_, c_ = loo_lda(X, rng.permutation(lab)); pa.append(a_); pc.append(H - c_)
        r.update(lda_acc_perm=float(np.mean(pa)), I_ce_perm=float(np.mean(pc)), chance=float((cnt / n).max()))
        lr = logreg_cv(X, lab)
        if lr: r.update(lr_acc=lr[0], I_ce_lr=H - lr[1], I_fano_lr=fano_bits(H, lr[0], S))
        # spectrum of between-speaker covariance
        lam, lam_raw = g["lam"], g["lam_raw"]
        r.update(lam=lam.tolist(), lam_raw=lam_raw.tolist(), ev_plug=g["ev"].tolist(),
                 pr_snr=pr(lam), pr_raw=pr(lam_raw), n_pos=int((lam > 0).sum()), n_snr1=int((lam > 1).sum()),
                 n_dirs_90=int(np.searchsorted(np.cumsum(np.log2(1 + np.clip(lam, 0, None))) / max(g["I_vc"], 1e-9), 0.9) + 1),
                 alpha_snr=powerlaw(lam, 16)[0], alpha_raw=powerlaw(lam_raw, 16)[0], alpha_k=powerlaw(lam, 16)[1],
                 trW=float(np.trace(g["W"])), trSb=float(np.clip(lam_raw, 0, None).sum()), sig2_tr=float(sg.mean(0).sum()))
        # naive between covariance of the speaker means (biased by W/n_s)
        _, _, _, M, _, _ = scatter(X, lab)
        r["pr_naive"] = pr(np.linalg.eigvalsh(np.cov(M.T)))
        # bootstrap over speakers
        bs = {k: [] for k in ["I_vc", "I_plug", "pr_snr", "pr_raw", "alpha_snr", "alpha_raw", "n_snr1"]}
        idx_by = [np.where(lab == s)[0] for s in u]
        for _ in range(NBOOT):
            pick = rng.integers(0, S, S)
            ii = np.concatenate([idx_by[j] for j in pick]); ll = np.concatenate([np.full(len(idx_by[j]), t) for t, j in enumerate(pick)])
            if len(ii) - S <= 34: continue
            try: gb = gauss_estimates(X[ii], ll)
            except Exception: continue
            bs["I_vc"].append(gb["I_vc"]); bs["I_plug"].append(gb["I_plug"]); bs["pr_snr"].append(pr(gb["lam"])); bs["pr_raw"].append(pr(gb["lam_raw"]))
            bs["alpha_snr"].append(powerlaw(gb["lam"], 16)[0]); bs["alpha_raw"].append(powerlaw(gb["lam_raw"], 16)[0]); bs["n_snr1"].append(int((gb["lam"] > 1).sum()))
        for k, v in bs.items():
            v = np.array(v, float); v = v[np.isfinite(v)]
            r[k + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 20 else None
        # ---------- intrinsic dimension of speaker means
        if S >= 20:
            L = np.linalg.cholesky(g["W"]); Mw = np.linalg.solve(L, M.T).T
            r.update(twonn_raw=twonn(M), twonn_white=twonn(Mw), lb_raw=levina_bickel(M), lb_white=levina_bickel(Mw))
            # calibration: Gaussian with same covariance and same number of points; and isotropic references
            C = np.cov(M.T); cal = [twonn(rng.multivariate_normal(np.zeros(32), C, S)) for _ in range(100)]
            call = [levina_bickel(rng.multivariate_normal(np.zeros(32), C, S)) for _ in range(100)]
            r.update(twonn_gauss=float(np.mean(cal)), twonn_gauss_sd=float(np.std(cal)), lb_gauss=float(np.mean(call)))
            for dd in (4, 8, 16):
                r[f"twonn_iso{dd}"] = float(np.mean([twonn(rng.standard_normal((S, dd))) for _ in range(100)]))
            # noise-only reference: within-speaker noise of the means
            Sn = g["W"] / (n / S); r["twonn_noise_only"] = float(np.mean([twonn(rng.multivariate_normal(np.zeros(32), Sn, S)) for _ in range(50)]))
            tb = []
            for _ in range(NBOOT):   # subsample 80% of speakers (bootstrap w/ duplicates breaks NN estimators)
                pick = rng.choice(S, int(0.8 * S), replace=False); tb.append(twonn(M[pick]))
            r["twonn_raw_ci"] = [float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))]
    # ---------- default voice vs LibriTTS (binary) and all speakers including LJ
    Xa, la = mu[tr], np.where(spk[tr] == "lj", "lj", "lib")
    cb = np.array([(la == "lj").sum(), (la == "lib").sum()]); Hb = entropy_bits(cb)
    lr = logreg_cv(Xa, la); r.update(Hbin=Hb, bin_acc=lr[0], I_bin_ce=Hb - lr[1], bin_chance=float(cb.max() / cb.sum()))
    labp = rng.permutation(la); lrp = logreg_cv(Xa, labp); r.update(I_bin_ce_perm=Hb - lrp[1])
    gb = gauss_estimates(Xa, la); r.update(I_bin_plug=gb["I_plug"], d_lj_lib=float(np.sqrt(max(gb["ev"][0] - 1, 0))))
    # all speakers incl. lj, speakers with >=2 utts
    lab_all = spk[tr]; ua, ca = np.unique(lab_all, return_counts=True); k2 = np.isin(lab_all, ua[ca >= 2])
    Xall, lall = mu[tr][k2], lab_all[k2]; ua, ca = np.unique(lall, return_counts=True); Hall = entropy_bits(ca)
    r.update(Hall=Hall, Sall=len(ua))
    if len(Xall) - len(ua) > 40:
        acc, ce = loo_lda(Xall, lall); ga = gauss_estimates(Xall, lall)
        r.update(all_acc=acc, I_all_ce=Hall - ce, I_all_fano=fano_bits(Hall, acc, len(ua)), I_all_plug=ga["I_plug"])
    res[run] = r
    print(run, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, list)}, flush=True)
json.dump(res, open(os.path.join(out, "speaker.json"), "w"), indent=1)
