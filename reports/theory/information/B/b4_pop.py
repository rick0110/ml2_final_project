"""b4: population model. Calibrates s2 (and optionally M_eff) ONLY on the trained voices q_seen(K) and predicts new voices.
Usage: python3 -I b4_pop.py out/b0.npz out/ <run_ref> [nrep]"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from bt_lib import load
from bt_pop import *
S = load(sys.argv[1]); OUT = sys.argv[2]; RUN = sys.argv[3]; NREP = int(sys.argv[4]) if len(sys.argv) > 4 else 30
p = Pop(S[RUN]); rng0 = np.random.default_rng(0)
# observed (mean over plain/wide runs N>=8): q_seen per K; external q per N (K=32)
Q_SEEN = {8: np.mean([.55, .55, .42, .49, .48]), 32: np.mean([.26, .22, .20, .24, .24, .22, .34, .33]), 128: np.mean([.09, .12, .14, .15])}
Q_EXT = {8: .434, 32: .302, 128: .211}

def sim(N, K, s2, logM, n_test=400, reps=NREP, seed=1, link="mean", R=1, kind="unseen", P=None, Mscale=1.0, post=True, per_voice=False):
    rng = np.random.default_rng(seed); out = []; pv = []
    for _ in range(reps):
        V = p.draw(N, rng)
        if kind == "unseen":
            vs = p.draw(n_test, rng)
        else:
            vs = V[rng.integers(0, N, n_test)]
        C = vs + p.noise(n_test, rng, R)
        Vhat = V + p.noise(N, rng, K)                        # class centre estimated from K utterances
        L = Lam(p, C, Vhat, K, s2, logM + np.log(Mscale), P=P, post=post)
        pr = pc(L, link); out.append(1 - pr.mean()); pv.append(pr)
    if per_voice: return float(np.mean(out)), np.concatenate(pv)
    return float(np.mean(out))

def fit(link, freeM, post=True):
    def loss(th):
        s2 = np.exp(th[0]); lm = th[1] if freeM else np.log(2000)
        return sum((sim(32, K, s2, lm, reps=8, link=link, kind="seen", post=post) - q) ** 2 for K, q in Q_SEEN.items())
    best = None
    for ls in np.linspace(0, 5, 11):
        for lm in ([np.log(2000), np.log(13100), np.log(50000)] if freeM else [np.log(2000)]):
            v = loss([ls, lm])
            if best is None or v < best[0]: best = (v, ls, lm)
    if freeM:
        o = minimize(loss, [best[1], best[2]], method="Nelder-Mead", options=dict(xatol=1e-2, fatol=1e-5, maxiter=200)); th = o.x; v = o.fun
    else:
        o = minimize_scalar(lambda x: loss([x, 0]), bounds=(best[1] - 0.5, best[1] + 0.5), method="bounded"); th = [o.x, np.log(2000)]; v = o.fun
    return float(np.exp(th[0])), float(th[1]), float(v)

res = dict(run=RUN, Q_SEEN=Q_SEEN, Q_EXT=Q_EXT, models={})
for link in ["mean", "samp"]:
    for freeM in [False, True]:
        s2, lm, v = fit(link, freeM)
        key = f"{link}_freeM{int(freeM)}"
        r = dict(s2=s2, Meff=float(np.exp(lm)), sse_seen=v)
        r["seen_K"] = {K: sim(32, K, s2, lm, link=link, kind="seen") for K in [8, 32, 128]}
        r["seen_N"] = {N: sim(N, 32, s2, lm, link=link, kind="seen") for N in [8, 32, 128, 512]}
        r["unseen_N"] = {N: sim(N, 32, s2, lm, link=link, reps=(NREP if N <= 512 else 6)) for N in [2, 8, 32, 128, 512, 2048]}
        r["unseen_K_at_N32"] = {K: sim(32, K, s2, lm, link=link) for K in [8, 32, 128]}
        r["unseen_halfM"] = {N: sim(N, 32, s2, lm, link=link, Mscale=0.5) for N in [32, 128]}
        r["unseen_2K"] = {N: sim(N, 64, s2, lm, link=link) for N in [32, 128]}
        r["unseen_R4"] = {N: sim(N, 32, s2, lm, link=link, R=4) for N in [8, 32, 128]}
        # per-voice distribution at N=8,32,128: fraction of voices with 0.1<p<0.9 (25 utterances: binomial not included)
        r["frac_mid"] = {}
        for N in [8, 32, 128]:
            _, pv = sim(N, 32, s2, lm, link=link, per_voice=True, reps=10)
            r["frac_mid"][N] = float(np.mean((pv > .1) & (pv < .9)))
        res["models"][key] = r
        print(key, json.dumps(r), flush=True)
json.dump(res, open(os.path.join(OUT, f"b4_{RUN}.json"), "w"), indent=1)
