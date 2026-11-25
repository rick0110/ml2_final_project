"""b26: extrapolacao de C(N) (38 externas, EC-LOVO, centroide) com bootstrap pareado de vozes; N* para alcancar a fala real.
Uso: python3 -I b26_extrap.py out/b12_emb.npz [B]"""
import sys
import numpy as np
from scipy.special import logsumexp
d = np.load(sys.argv[1], allow_pickle=False); B = int(sys.argv[2]) if len(sys.argv) > 2 else 2000; rng = np.random.default_rng(13)
keys = d["keys"].astype(str); E = d["emb"] / np.linalg.norm(d["emb"], axis=1, keepdims=True); K_ = [k.split("|") for k in keys]
ref = sorted(k[2] for k in K_ if k[0] == "ref"); oth = {}
for i, k in enumerate(K_):
    if k[0] == "oth": oth.setdefault(k[1], []).append(E[i])
C = np.array([np.mean(oth[r.split("_")[0]], 0) for r in ref]); C /= np.linalg.norm(C, axis=1, keepdims=True)
BET = np.linspace(1, 150, 150)
def tab(sim): return np.array([[-(b * sim[i, i] - logsumexp(b * sim[i])) / np.log(2) for i in range(len(sim))] for b in BET])
def ce(T, ii):
    t = T[:, ii]; tot = t.sum(1); return np.log2(38) - np.mean([t[np.argmin(tot - t[:, j]), j] for j in range(len(ii))])
def outs(run):
    m = {k[3].rsplit("__", 1)[0]: E[i] for i, k in enumerate(K_) if k[0] == "out" and k[1] == run and k[2] == "strength_1.0"}; return np.array([m[r] for r in ref])
T8, T128 = tab(outs("n8_k32_s1-wide") @ C.T), tab(outs("n128_k32_s1-wide") @ C.T)
real = np.array([oth[r.split("_")[0]][0] for r in ref]); Cm = np.array([np.mean(oth[r.split("_")[0]][1:], 0) for r in ref]); Cm /= np.linalg.norm(Cm, axis=1, keepdims=True)
TR = tab(real @ Cm.T)
full = np.arange(38); c8, c128, cr = ce(T8, full), ce(T128, full), ce(TR, full)
bs = []
for _ in range(B):
    ii = rng.integers(0, 38, 38); a, b, r = ce(T8, ii), ce(T128, ii), ce(TR, ii); s = (b - a) / 4
    nstar = 128 * 2 ** ((r - b) / s) if s > 0 else np.inf
    bs.append([s, b, b + 2 * s, b + 4 * s, np.log10(nstar), r])
bs = np.array(bs); s0 = (c128 - c8) / 4
q = lambda j: np.percentile(bs[:, j], [2.5, 50, 97.5])
print(f"C(8)={c8:.2f} C(128)={c128:.2f} fala real={cr:.2f} bits (EC, 38 vozes, teto 5.25)")
print(f"inclinacao s = {s0:.3f} bit/duplicacao; IC95 {q(0)[0]:.3f}..{q(0)[2]:.3f}")
print(f"C(512) prev = {c128+2*s0:.2f} [{q(2)[0]:.2f}, {q(2)[2]:.2f}];  C(2048) prev = {c128+4*s0:.2f} [{q(3)[0]:.2f}, {q(3)[2]:.2f}] (log-linear; teto 5.25)")
nst = 128 * 2 ** ((cr - c128) / s0)
print(f"N* (saida = fala real, {cr:.2f} bits) = {nst:.3g}; IC95 10^{q(4)[0]:.1f} .. 10^{q(4)[2]:.1f} locutores")
