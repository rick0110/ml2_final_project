"""Tasks 2-3 add-ons: parametric-bootstrap bias correction of the Gaussian speaker information, split-half estimate,
bits captured by top-k directions, predicted gain of averaging R references, denoised intrinsic dimension,
empirical covering exponent inside one latent space.
usage: python3 -I info_speaker2.py DATA OUT [nsim]"""
import sys, os, json, warnings; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from info_common import *
from info_speaker_lib import scatter, gauss_estimates, twonn, levina_bickel
warnings.filterwarnings("ignore")
root, out = sys.argv[1], sys.argv[2]
NSIM = int(sys.argv[3]) if len(sys.argv) > 3 else 100
rng = np.random.default_rng(1)

def psd(A):
    w, V = np.linalg.eigh((A + A.T) / 2); return (V * np.clip(w, 0, None)) @ V.T

def simulate(Sb, W, cnt):
    p = W.shape[0]; S = len(cnt)
    v = rng.multivariate_normal(np.zeros(p), Sb, S)
    lab = np.repeat(np.arange(S), cnt)
    X = v[lab] + rng.multivariate_normal(np.zeros(p), W, len(lab))
    return X, lab

def info_from(Sb, W, R=1):
    lam = np.clip(eigh(psd(Sb), W, eigvals_only=True), 0, None)
    return 0.5 * np.log2(1 + R * lam).sum(), np.sort(lam)[::-1]

def split_half(X, lab):
    """unbiased between-cov from cross-covariance of two disjoint halves of each speaker's utterances"""
    u = np.unique(lab); A, B = [], []
    for s in u:
        idx = rng.permutation(np.where(lab == s)[0]); h = len(idx) // 2
        if h < 1: continue
        A.append(X[idx[:h]].mean(0)); B.append(X[idx[h:2 * h]].mean(0))
    A, B = np.array(A), np.array(B); A -= A.mean(0); B -= B.mean(0)
    C = (A.T @ B + B.T @ A) / (2 * (len(A) - 1))
    return C

def covering_curve(Mw_denoised, ms, reps=200):
    """median NN distance from a held-out speaker to the nearest of m other speakers (whitened, denoised means)"""
    S = len(Mw_denoised); D = cdist(Mw_denoised, Mw_denoised); np.fill_diagonal(D, np.inf); res = []
    for m in ms:
        vals = []
        for _ in range(reps):
            a = rng.integers(S); others = rng.choice(np.delete(np.arange(S), a), m, replace=False)
            vals.append(D[a, others].min())
        res.append(float(np.median(vals)))
    return np.array(res)

def slope(ms, d):
    return float(-np.polyfit(np.log(ms), np.log(d), 1)[0])

res = {}
for run, N, K, seed, var in list_runs(root):
    if var not in ("", "wide"): continue
    d = load_latents(root, run); tr = d["train"]; spk = d["speaker"]; lib = tr & (spk != "lj")
    X, lab, sg = d["mu"][lib], spk[lib], np.exp(d["logvar"][lib])
    u, cnt = np.unique(lab, return_counts=True); keep = np.isin(lab, u[cnt >= 2])
    X, lab, sg = X[keep], lab[keep], sg[keep]
    u, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True); S = len(u); n = len(X)
    if n - S <= 40: continue
    g = gauss_estimates(X, lab); W = g["W"]; Sb = psd(g["Sb"]); noise = np.diag(sg.mean(0))
    I0, lam = info_from(Sb, W); I0c, lamc = info_from(Sb, W + noise)
    r = dict(run=run, N=N, K=K, seed=seed, variant=var, S=S, n=n, I_vc=I0, I_vc_chan=I0c, log2S=float(np.log2(S)))
    # parametric bootstrap: data generated from the fitted model, same design; bias = E[I_hat] - I_model
    sims, simsc, simpr = [], [], []
    for _ in range(NSIM):
        Xs, ls = simulate(Sb, W, cnt)
        gs = gauss_estimates(Xs, ls); sims.append(gs["I_vc"])
        simsc.append(info_from(psd(gs["Sb"]), gs["W"] + noise)[0]); simpr.append(pr(np.linalg.eigvalsh(psd(gs["Sb"]))))
    sims, simsc = np.array(sims), np.array(simsc)
    r.update(bias=float(sims.mean() - I0), bias_chan=float(simsc.mean() - I0c),
             I_bc=float(2 * I0 - sims.mean()), I_bc_chan=float(2 * I0c - simsc.mean()),
             I_bc_sd=float(sims.std()), I_bc_chan_sd=float(simsc.std()),
             pr_raw=pr(np.linalg.eigvalsh(Sb)), pr_raw_bias=float(np.mean(simpr) - pr(np.linalg.eigvalsh(Sb))))
    # split-half (noise-free between covariance), average over 20 random splits
    Cs = np.mean([split_half(X, lab) for _ in range(20)], 0)
    r["I_split"] = info_from(Cs, W)[0]; r["I_split_chan"] = info_from(Cs, W + noise)[0]
    # cumulative bits by number of directions (generalised eigen-directions of Sb vs W, channel version)
    cb = np.cumsum(0.5 * np.log2(1 + lamc)); cbm = np.cumsum(0.5 * np.log2(1 + lam))
    r["cum_bits_chan"] = cb.tolist(); r["cum_bits_mu"] = cbm.tolist(); r["snr_chan"] = lamc.tolist()
    for q in (0.5, 0.9, 0.95, 0.99):
        r[f"k{int(q*100)}_chan"] = int(np.searchsorted(cb / cb[-1], q) + 1)
        r[f"k{int(q*100)}_mu"] = int(np.searchsorted(cbm / cbm[-1], q) + 1)
    # averaging R references: gain predicted by the Gaussian model (noise W/R; posterior noise not averaged)
    for R in (2, 4, 16):
        r[f"gain_R{R}_mu"] = float(0.5 * np.log2((1 + R * lam) / (1 + lam)).sum())
        lamR = np.clip(eigh(Sb, W / R + noise, eigvals_only=True), 0, None)
        r[f"gain_R{R}_chan"] = float(0.5 * np.log2(1 + lamR).sum() - I0c)
    # ---------- intrinsic dimension of denoised speaker means (only populations with S >= 30)
    if S >= 30:
        M = np.zeros((S, X.shape[1])); np.add.at(M, inv, X); M /= cnt[:, None]; m0 = X.mean(0)
        L = np.linalg.cholesky(W); Linv = np.linalg.inv(L)
        # BLUP shrinkage of each mean toward the centre: E[v|M_s] = Sb (Sb + W/n_s)^-1 (M_s - m0)
        Md = np.array([Sb @ np.linalg.solve(Sb + W / cnt[s] + 1e-9 * np.eye(32), M[s] - m0) for s in range(S)])
        Mdw = Md @ Linv.T
        r["twonn_denoised"] = twonn(Mdw); r["lb_denoised"] = levina_bickel(Mdw, 5)
        Sbw = Linv @ Sb @ Linv.T
        r["pr_Sbw"] = pr(np.linalg.eigvalsh(Sbw))
        # null: a Gaussian with the same between covariance and same S, passed through the same shrinkage
        null_t, null_l = [], []
        for _ in range(100):
            Xs, ls = simulate(Sb, W, cnt); us, invs, cs = np.unique(ls, return_inverse=True, return_counts=True)
            Ms = np.zeros((len(us), 32)); np.add.at(Ms, invs, Xs); Ms /= cs[:, None]
            Mds = np.array([Sb @ np.linalg.solve(Sb + W / cs[s] + 1e-9 * np.eye(32), Ms[s] - Xs.mean(0)) for s in range(len(us))]) @ Linv.T
            null_t.append(twonn(Mds)); null_l.append(levina_bickel(Mds, 5))
        r.update(twonn_null=float(np.mean(null_t)), twonn_null_ci=[float(np.percentile(null_t, 2.5)), float(np.percentile(null_t, 97.5))],
                 lb_null=float(np.mean(null_l)))
        sub = [twonn(Mdw[rng.choice(S, int(0.8 * S), replace=False)]) for _ in range(200)]
        subl = [levina_bickel(Mdw[rng.choice(S, int(0.8 * S), replace=False)], 5) for _ in range(200)]
        r["twonn_denoised_ci"] = [float(np.percentile(sub, 2.5)), float(np.percentile(sub, 97.5))]
        r["lb_denoised_ci"] = [float(np.percentile(subl, 2.5)), float(np.percentile(subl, 97.5))]
        # PCA spectrum of the (whitened) between covariance with bootstrap over speakers
        ev = np.sort(np.linalg.eigvalsh(Sbw))[::-1]; r["eig_Sbw"] = ev.tolist()
        # covering curve: NN distance vs number of trained speakers, inside this latent space
        if S >= 100:
            ms = np.array([4, 8, 16, 32, 64, S - 1])
            dc = covering_curve(Mdw, ms); r["cover_ms"] = ms.tolist(); r["cover_d"] = dc.tolist()
            r["cover_slope"] = slope(ms[:-1], dc[:-1]); r["cover_dim"] = 1 / r["cover_slope"]
            # same curve for Gaussian null (same Sb, same S) -> finite-sample dimension a linear Gaussian would show
            nulls = []
            for _ in range(20):
                v = rng.multivariate_normal(np.zeros(32), Sbw, S); nulls.append(slope(ms[:-1], covering_curve(v, ms, 100)[:-1]))
            r["cover_slope_null"] = float(np.mean(nulls)); r["cover_slope_null_sd"] = float(np.std(nulls))
            bs = []
            for _ in range(100):
                pick = rng.choice(S, S, replace=True); pick = np.unique(pick)
                if len(pick) < 70: continue
                mm = np.array([4, 8, 16, 32, 64]); bs.append(slope(mm, covering_curve(Mdw[pick], mm, 100)))
            r["cover_slope_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res[run] = r
    print(run, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, list)}, flush=True)
json.dump(res, open(os.path.join(out, "speaker2.json"), "w"), indent=1)
