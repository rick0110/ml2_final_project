"""b14: modelo de canal com capacidade limitada (CAP), sem parametro ajustado:
 - largura de decodificacao s2(N) tirada da capacidade MEDIDA no audio, C(N) (b13, limite CE): 1/2 sum log2(1+lambda_j/s2) = C(N)
 - peso a priori da voz padrao derivado da exposicao: M_eff = M + (passos_en*lote)/(apresentacoes por frase no ajuste) = 2M + NK
 - ligacao: media posterior + Lema 1 (Phi(20.8(1/2-rho)))  [ou estado com tau do ICC, variante]
Uso: python3 -I b14_cap.py out/b0.npz [Cmode]   Cmode = ce (padrao) | fano"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import norm, spearmanr
from scipy.special import logsumexp
from scipy.optimize import brentq
from bt_lib import load, Geo
S = load(sys.argv[1]); MODE = sys.argv[2] if len(sys.argv) > 2 else "ce"
CM = {"ce": (0.75, 2.01), "fano": (0.31, 1.13)}[MODE]                  # C(8), C(128) medidos (centroide/frase-ref medios)
def C_of(N): return CM[0] + (CM[1] - CM[0]) * np.log2(N / 8) / 4          # interpolacao log-linear (N=32: INTERPOLADO)
g = Geo(S["n128_k32_s1-wide"]); Wr = g.W + 0.05 * np.trace(g.W) / 32 * np.eye(32); Li = np.linalg.inv(np.linalg.cholesky(Wr))
LAM = np.clip(np.linalg.eigvalsh(Li @ g.Sb @ Li.T), 0, None)
def s2_of(C): return brentq(lambda s: 0.5 * np.log2(1 + LAM / s).sum() - C, 1e-3, 1e6)
def Meff(M, N, K, steps_en=2500 * 48, steps_ft=2500 * 48):
    r = steps_ft / (M + N * K); return M + steps_en / r
A = 0.75 / 0.036
def link(L, kind="lema"):
    rho = 1 / (1 + np.exp(np.clip(L, -50, 50)))
    return norm.cdf(A * (0.5 - rho))
def geo(r):
    g = Geo(S[r]); Wr = g.W + 0.05 * np.trace(g.W) / 32 * np.eye(32); Li = np.linalg.inv(np.linalg.cholesky(Wr))
    Mw = g.means @ Li.T; v0 = Li @ g.mlj; trW = np.trace(Li @ g.W @ Li.T); Cw = g.C @ Li.T
    d0 = ((Cw - v0) ** 2).sum(1); d2 = ((Cw[:, None] - Mw[None]) ** 2).sum(-1) - trW / g.n[None, :]
    return g, d0, d2, Cw, Mw, v0, trW
def Lam(r, M=2000, scale=1.0, C=None, theta=0.0, own_only=False):
    g, d0, d2, Cw, Mw, v0, trW = geo(r)
    if scale != 1.0:
        Cs = scale * Cw; d0 = ((Cs - v0) ** 2).sum(1); d2 = ((Cs[:, None] - Mw[None]) ** 2).sum(-1) - trW / g.n[None, :]
    s2 = s2_of(C if C is not None else C_of(g.N))
    return np.log(g.K) + logsumexp(-d2 / (2 * s2), axis=1) - np.log(Meff(M, g.N, g.K)) + d0 / (2 * s2) - theta, g, s2
def obs(g, grp, key="r1"):
    idx = [j for j, x in enumerate(g.refs) if x["grp"] == grp and key in x]
    return np.array(idx), np.array([g.refs[j][key]["k"] / g.refs[j][key]["n"] for j in idx])
res = {"mode": MODE}
print(f"C(N) modo {MODE}: C(8)={C_of(8):.2f} C(32)={C_of(32):.2f} (interp.) C(128)={C_of(128):.2f}; s2: " + " ".join(f"{s2_of(C_of(n)):.1f}" for n in [8, 32, 128, 200]))
print("M_eff = 2M+NK: N=8 %d, N=32 %d, N=128 %d (K=32)" % tuple(Meff(2000, n, 32) for n in [8, 32, 128]))
rng = np.random.default_rng(2)
def table(theta=0.0, tag=""):
    out = {}
    for r in ["n8_k32_s1-wide", "n32_k32_s1-wide", "n128_k32_s1-wide"]:
        L, g, s2 = Lam(r, theta=theta); idx, p = obs(g, "external"); pr = link(L[idx])
        sim = rng.binomial(25, np.tile(pr, (300, 1))) / 25
        out[r] = dict(q_obs=float(1 - p.mean()), q_pred=float(1 - pr.mean()), rho=float(spearmanr(L[idx], p).correlation),
                      mid_obs=float(np.mean((p > .1) & (p < .9))), mid_pred=float(np.mean((sim > .1) & (sim < .9))))
        o = out[r]; print(f"  {tag}{r}: q obs {o['q_obs']:.2f} prev {o['q_pred']:.2f} | Spearman {o['rho']:.2f} | vozes 10-90% obs {o['mid_obs']:.2f} prev {o['mid_pred']:.2f}")
    return out
print("\n(A) zero parametros ajustados")
res["A"] = table()
# (B) one offset fitted on N=128 only
def q128(th):
    L, g, _ = Lam("n128_k32_s1-wide", theta=th); idx, p = obs(g, "external"); return 1 - link(L[idx]).mean() - (1 - p.mean())
th = brentq(q128, -30, 30); print(f"\n(B) 1 parametro: theta (log-prior extra) ajustado SO em N=128 = {th:.2f} nats")
res["B_theta"] = th; res["B"] = table(th, "")
print("\n(C) treinadas na grade (theta de B): q obs vs prev por N e K")
rows = []
for r in sorted(S):
    if S[r]["var"] not in ("", "wide") or not S[r]["refs"] or S[r]["N"] < 8: continue
    L, g, s2 = Lam(r, theta=th); idx, p = obs(g, "seen")
    rows.append((g.N, g.K, 1 - p.mean(), 1 - link(L[idx]).mean()))
rows = np.array(rows)
for key, col in [("N", 0), ("K", 1)]:
    for v in sorted(set(rows[:, col])):
        m = rows[:, col] == v; print(f"  {key}={int(v):3d}: obs {rows[m,2].mean():.3f} prev {rows[m,3].mean():.3f} (n={m.sum()})")
res["seen_rows"] = rows.tolist()
print("\n(D) forca 0.6 (treinos plain; s2 do N do treino; codigo x 0.6) e M/2")
agg = {}
for r in sorted(S):
    if S[r]["var"] != "" or not S[r]["refs"] or S[r]["N"] < 8: continue
    L1, g, _ = Lam(r, theta=th); L6, _, _ = Lam(r, theta=th, scale=0.6)
    for j, x in enumerate(g.refs):
        if "s06" not in x: continue
        a = agg.setdefault(x["grp"], [[], [], [], []]); a[0].append(x["r1"]["k"] / 25); a[1].append(x["s06"]["k"] / 25); a[2].append(link(L1[j])); a[3].append(link(L6[j]))
for k_, a in agg.items():
    a = np.array(a); print(f"  {k_:8s}: copia 1.0 obs {a[0].mean():.2f} prev {a[2].mean():.2f} | 0.6 obs {a[1].mean():.2f} prev {a[3].mean():.2f}")
L, g, _ = Lam("n128_k32_s1-wide", theta=th); idx, p = obs(g, "external")
L2, _, _ = Lam("n128_k32_s1-wide", theta=th, M=1000)
print(f"  M=1000 (N=128): q {1-link(L[idx]).mean():.3f} -> {1-link(L2[idx]).mean():.3f};  Delta log-prior = {np.log(Meff(2000,128,32)/Meff(1000,128,32))*1:.2f} nats (contagem ingenua: {np.log(2):.2f})")
for N in [512, 2048]:
    # extrapolate: C(N) log-linear; trained set grows -> use population draw? here: same geometry, only s2 and M_eff change (coverage fixed: conservative)
    s2 = s2_of(C_of(N)); gg, d0, d2, *_ = geo("n128_k32_s1-wide")
    Lx = np.log(32) + logsumexp(-d2 / (2 * s2), axis=1) + np.log(N / 128) - np.log(Meff(2000, N, 32)) + d0 / (2 * s2) - th
    print(f"  N={N}: C={C_of(N):.2f} bits, s2={s2:.1f}, q prev (cobertura escalada por N/128) {1-link(Lx[idx]).mean():.3f}")
json.dump(res, open(f"out/b14_{MODE}.json", "w"), indent=1, default=float)
