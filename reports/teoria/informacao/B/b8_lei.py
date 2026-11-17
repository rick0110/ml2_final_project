"""b8: lei de razao de verossimilhanca 'atomo + cobertura' em unidades de log-contagem.
Treinadas: Lambda = ln K - ln M_eff + (d0^2 - d_own^2)/(2 s^2)          (atomo proprio, sem emprestimo: K so)
Novas:     Lambda = ln K - ln M_eff + d0^2/(2 s^2) + g * LSE_i(-d_i^2/(2 s^2))   (cobertura; g=1 NW; versao 'vizinho' usa min)
s^2 = 12.1 (dispersao das treinadas, b5), tau = 0.83 (ICC), ln M_eff: UNICO parametro ajustado, nas treinadas.
Uso: python3 -I b8_lei.py out/b0.npz"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import norm, spearmanr
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
from bt_lib import load, Geo
S = load(sys.argv[1]); S2 = 12.1; TAU = 0.83
RUNS = sorted([r for r in S if S[r]["var"] in ("", "wide") and S[r]["refs"] and S[r]["N"] >= 8], key=lambda r: (S[r]["N"], S[r]["K"], r))
G = {}
for r in RUNS:
    g = Geo(S[r]); Wreg = g.W + 0.05 * np.trace(g.W) / 32 * np.eye(32); Li = np.linalg.inv(np.linalg.cholesky(Wreg))
    Mw = g.means @ Li.T; v0 = Li @ g.mlj; trW = np.trace(Li @ g.W @ Li.T)
    Cw = g.C @ Li.T
    d0 = ((Cw - v0) ** 2).sum(1) - trW
    d2 = ((Cw[:, None] - Mw[None]) ** 2).sum(-1) - trW * (1 + 1 / g.n[None, :])
    own = np.array([d2[j, g.ids.index(x["spk"])] if x["spk"] in g.ids else np.nan for j, x in enumerate(g.refs)])
    G[r] = dict(g=g, d0=d0, d2=d2, own=own)
def lam(r, lnM, kind, mode="nw"):
    D = G[r]; g = D["g"]
    if kind == "seen": ev = (D["d0"] - D["own"]) / (2 * S2)
    elif mode == "nw": ev = D["d0"] / (2 * S2) + logsumexp(-D["d2"] / (2 * S2), axis=1)
    else: ev = D["d0"] / (2 * S2) - D["d2"].min(1) / (2 * S2)
    return np.log(g.K) - lnM + ev
def q_of(r, grp, lnM, mode="nw"):
    g = G[r]["g"]; idx = [j for j, x in enumerate(g.refs) if x["grp"] == grp]
    if not idx: return None
    L = lam(r, lnM, "seen" if grp == "seen" else "unseen", mode)[idx]
    k = np.array([g.refs[j]["r1"]["k"] for j in idx]); n = np.array([g.refs[j]["r1"]["n"] for j in idx])
    return 1 - k.sum() / n.sum(), 1 - norm.cdf(L / TAU).mean(), L, k / n
def loss(lnM):
    return sum((q_of(r, "seen", lnM)[1] - q_of(r, "seen", lnM)[0]) ** 2 for r in RUNS)
o = minimize_scalar(loss, bounds=(0, 40), method="bounded"); lnM = o.x
print(f"ln M_eff ajustado nas treinadas = {lnM:.2f}  (M_eff = {np.exp(lnM):.3g}; ln 2000 = {np.log(2000):.2f}; ln 15100 = {np.log(15100):.2f})")
res = dict(lnM=lnM, rows=[])
for r in RUNS:
    g = G[r]["g"]; s = f"{r:18s} N={g.N:3d} K={g.K:3d}"; row = dict(run=r, N=g.N, K=g.K)
    for grp in ["seen", "unseen", "external"]:
        a = q_of(r, grp, lnM)
        if a is None: continue
        b = q_of(r, grp, lnM, mode="nn") if grp != "seen" else None
        s += f" | {grp[:3]} obs {a[0]:.2f} prev {a[1]:.2f}" + (f" (viz {b[1]:.2f})" if b else "")
        row[grp] = [a[0], a[1]] + ([b[1]] if b else [])
        if grp == "external" and len(a[3]) > 10: s += f" rho {spearmanr(a[2], a[3]).correlation:.2f}"
    res["rows"].append(row); print(s)
# seen q vs K / N summary
import collections
agg = collections.defaultdict(list)
for row in res["rows"]:
    if "seen" in row: agg[("K", row["K"])].append(row["seen"]); agg[("N", row["N"])].append(row["seen"])
for key in sorted(agg): a = np.array(agg[key]); print(f"  treinadas {key}: obs {a[:,0].mean():.3f} prev {a[:,1].mean():.3f} (n={len(a)})")
json.dump(res, open("out/b8.json", "w"), indent=1, default=float)
