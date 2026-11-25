"""b2: fits the kernel/posterior decoder only on the TRAINED (seen) voices of the grid and predicts the new and external ones.
Families: class covariance = W_c + Sigma_post,c + s2 * W_reg  (s2 = isotropic smoothing in the whitened space),
prior weight: K per LibriTTS voice, M_eff = exp(logM) for LJ.  Links: 'mean' (posterior mean + Lemma 1) and 'samp'.
Usage: python3 -I b2_ajuste.py out/b0.npz out/"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from bt_lib import *
S = load(sys.argv[1]); OUT = sys.argv[2]
BASE = sorted([r for r in S if S[r]["var"] in ("", "wide") and S[r]["refs"]], key=lambda r: (S[r]["N"], S[r]["K"], r))
GEO = {r: Geo(S[r]) for r in BASE}

def covs(g, s2, post=True):
    Wreg = g.W + 0.05 * np.trace(g.W) / 32 * np.eye(32)
    Slib = g.W + g.W / np.mean(g.n) + s2 * Wreg + (np.diag(g.s2lib) if post else 0)
    Slj = g.Wlj + s2 * Wreg + (np.diag(g.s2lj) if post else 0)
    return Slib, Slj

def lam_run(g, s2, logM, post=True, K=None, C=None, means=None):
    Slib, Slj = covs(g, s2, post)
    return lam(g, C=C, Slib=Slib, Slj=Slj, logM=logM, K=K, means=means)

def nll(theta, grp, link, post, runs):
    s2, logM = np.exp(theta[0]), theta[1]
    tot = 0.0
    for r in runs:
        g = GEO[r]; k, n, idx = obs(g, grp)
        if len(k) == 0: continue
        p = np.clip(p_copy(lam_run(g, s2, logM, post)[idx], link), 1e-4, 1 - 1e-4)
        tot -= (k * np.log(p) + (n - k) * np.log(1 - p)).sum()
    return tot

res = {}
for post in [True, False]:
    for link in ["mean", "samp"]:
        for freeM in [False, True]:
            key = f"post={int(post)}_link={link}_freeM={int(freeM)}"
            best = None
            for ls2 in np.linspace(-1, 6, 15):
                if freeM:
                    for lm in [np.log(2000), np.log(2000) + 3, np.log(2000) - 3]:
                        o = minimize(lambda t: nll(t, "seen", link, post, BASE), [ls2, lm], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-3, maxiter=400))
                        if best is None or o.fun < best.fun: best = o
                else:
                    f = nll([ls2, np.log(2000)], "seen", link, post, BASE)
                    if best is None or f < best["fun"]: best = dict(fun=f, x=np.array([ls2, np.log(2000)]))
            if not freeM:
                o = minimize(lambda t: nll([t[0], np.log(2000)], "seen", link, post, BASE), [best["x"][0]], method="Nelder-Mead")
                best = dict(fun=o.fun, x=np.array([o.x[0], np.log(2000)]))
            th = best["x"] if isinstance(best, dict) else best.x; fun = best["fun"] if isinstance(best, dict) else best.fun
            s2, logM = float(np.exp(th[0])), float(th[1])
            rows = []
            for r in BASE:
                g = GEO[r]; L = lam_run(g, s2, logM, post); row = dict(run=r, N=g.N, K=g.K)
                for grp in ["seen", "unseen", "external"]:
                    k, n, idx = obs(g, grp)
                    if len(k) == 0: continue
                    p = p_copy(L[idx], link)
                    row[grp] = (float(1 - k.sum() / n.sum()), float(1 - p.mean()))
                    if grp == "external" and len(k) > 10:
                        row["rho_ext"] = float(spearmanr(L[idx], k / n).correlation)
                        row["nll_ext_bits"] = float(-(k * np.log2(np.clip(p, 1e-4, 1)) + (n - k) * np.log2(np.clip(1 - p, 1e-4, 1))).sum() / n.sum())
                rows.append(row)
            res[key] = dict(s2=s2, logM=logM, Meff=float(np.exp(logM)), nll_seen=float(fun), rows=rows)
            print(f"\n== {key}: s2={s2:.2f} M_eff={np.exp(logM):.0f} nll_seen={fun:.1f}")
            for row in rows:
                s = f"{row['run']:18s}"
                for grp in ["seen", "unseen", "external"]:
                    if grp in row: s += f" | {grp[:3]} obs {row[grp][0]:.2f} pred {row[grp][1]:.2f}"
                if "rho_ext" in row: s += f" | rho {row['rho_ext']:.2f} nll {row['nll_ext_bits']:.2f}"
                print(s)
json.dump(res, open(os.path.join(OUT, "b2.json"), "w"), indent=1)
