"""b25: decomposicao nas 38 externas (wide): espalhamento das saidas (H(Y)) vs ruido por voz (proxy H(Y|S): R=1 x R=4 da mesma referencia),
e vs distancia ao centroide real (erro). Gaussiano em base PCA comum + escalar. Uso: python3 -I b25_decomp38.py out/b0.npz out/b12_emb.npz [B]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bt_lib import load
S = load(sys.argv[1]); d = np.load(sys.argv[2], allow_pickle=False); B = int(sys.argv[3]) if len(sys.argv) > 3 else 1000; rng = np.random.default_rng(9)
keys = d["keys"].astype(str); E = d["emb"] / np.linalg.norm(d["emb"], axis=1, keepdims=True); K_ = [k.split("|") for k in keys]
oth = {}
for i, k in enumerate(K_):
    if k[0] == "oth": oth.setdefault(k[1], []).append(E[i])
ext = sorted(x["ref"] for x in S["n128_k32_s1-wide"]["refs"] if x["grp"] == "external")
def get(run, cond): m = {k[3].rsplit("__", 1)[0]: E[i] for i, k in enumerate(K_) if k[0] == "out" and k[1] == run and k[2] == cond}; return np.array([m[v] for v in ext])
Cr = np.array([np.mean(oth[v.split("_")[0]], 0) for v in ext]); Cr /= np.linalg.norm(Cr, axis=1, keepdims=True)
Y = {N: (get(f"n{N}_k32_s1-wide", "strength_1.0"), get(f"n{N}_k32_s1-wide", "strength_1.0_r4")) for N in [8, 128]}
allY = np.vstack([Y[8][0], Y[8][1], Y[128][0], Y[128][1], Cr]); mu0 = allY.mean(0); Vt = np.linalg.svd(allY - mu0, full_matrices=False)[2]
def stats(A, Bm, Cr, idx):
    A, Bm, C = A[idx], Bm[idx], Cr[idx]
    Dw = np.mean(((A - Bm) ** 2).sum(1)) / 2; M = (A + Bm) / 2; Dtot = np.mean(((M - M.mean(0)) ** 2).sum(1))
    Derr = np.mean(((A - C) ** 2).sum(1)); Dref = np.mean(((C - C.mean(0)) ** 2).sum(1))
    out = [Dtot, Dw, Derr, Dref]
    for k in [3, 6]:
        P = Vt[:k]; a, b = (A - mu0) @ P.T, (Bm - mu0) @ P.T
        Sw = (a - b).T @ (a - b) / (2 * len(a)); St = np.cov(np.vstack([a, b]).T)
        out += [0.5 * np.log2(np.linalg.det(St) / np.linalg.det(Sw))]
        out += [0.5 * np.log2(np.linalg.det(2*np.pi*np.e*St)), 0.5 * np.log2(np.linalg.det(2*np.pi*np.e*Sw))]
    return np.array(out)
names = ["espalhamento entre vozes D_tot", "ruido por voz D_w (R1 x R4)", "erro saida-centroide real", "espalhamento das vozes reais",
         "I_G k=3", "H(Y) k=3", "H(Y|S) k=3", "I_G k=6", "H(Y) k=6", "H(Y|S) k=6"]
st = {N: stats(*Y[N], Cr, np.arange(38)) for N in [8, 128]}
bs = np.array([stats(*Y[128], Cr, ii) - stats(*Y[8], Cr, ii) for ii in (rng.integers(0, 38, 38) for _ in range(B))])
print("38 vozes externas, forca 1.0 (frase s0). Distancias quadradas em embeddings unitarios; entropias gaussianas em bits.")
for j, nm in enumerate(names):
    print(f"  {nm:34s}: N=8 {st[8][j]:7.3f}  N=128 {st[128][j]:7.3f}  dif {st[128][j]-st[8][j]:+.3f} [IC95 {np.percentile(bs[:,j],2.5):+.3f}, {np.percentile(bs[:,j],97.5):+.3f}]")
