"""t2: verificacao por simulacao da formula
   q(N,K,M) = E_c[ (1 - F_c(t(c)))^N ],  t(c) = ||c - v0||^2 - 2T ln(M/K),
contra (a) o canal de Gibbs simulado diretamente (livros-codigo aleatorios, temperatura T, massas M e K),
e (b) a aproximacao de espectro de informacao q ~ P_c[ L(c) > log2 N ], L(c) = -log2 F_c(t(c)).
Geometria gaussiana calibrada no run N=128 (Sb, centro, v0, ruido de referencia Wx).
python3 -I t2_verifica_formula.py <coverage2> <out>
"""
import sys, os, json
import numpy as np
from scipy.special import logsumexp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_lib import load_latents, Geometry, RUNS_WIDE
from rd_theory import gchisq_cdf_imhof

cov, out = sys.argv[1:3]
rng = np.random.default_rng(7)
G = Geometry(load_latents(f'{cov}/{RUNS_WIDE[128]}/step_2500/latents.npz'))
ev, U = np.linalg.eigh(G.Sb); ev = np.clip(ev, 0, None)
mu, v0, Wx = G.center, G.lj, G.Wx
Lv = U @ np.diag(np.sqrt(ev))
Lw = np.linalg.cholesky(Wx)
nc = 600
C = mu + rng.standard_normal((nc, 32)) @ Lv.T + rng.standard_normal((nc, 32)) @ Lw.T
rep = {}
for (T, K, lnM) in [(1.0, 32, np.log(32) + 116.5 / 2), (1.0, 8, np.log(32) + 116.5 / 2), (2.0, 32, np.log(32) + 116.5 / 4),
                    (1.0, 32, np.log(2000))]:
    tau = 2 * T * (lnM - np.log(K))
    key = f'T={T},K={K},tau={tau:.1f}'
    w = ev + 1.0 / K
    L = []
    F = []
    for c in C:
        a = U.T @ (c - mu)
        t = ((c - v0) ** 2).sum() - tau
        f = gchisq_cdf_imhof(t, w, a ** 2 / w) if t > 0 else 0.0
        F.append(f)
    F = np.array(F)
    row = {}
    for N in (8, 32, 128, 512):
        # formula
        qf = float(np.mean(np.exp(N * np.log1p(-np.minimum(F, 1 - 1e-15)))))
        with np.errstate(divide='ignore'):
            Lc = -np.log2(F)
        qis = float(np.mean(Lc > np.log2(N)))
        # simulacao direta do canal de Gibbs (decisao: massa da LJ > 1/2 -> nao copiada)
        nfail = 0; ntot = 0
        for rep_ in range(4):
            cb = mu + rng.standard_normal((N, 32)) @ Lv.T + rng.standard_normal((N, 32)) / np.sqrt(K)
            V = np.vstack([v0[None], cb])
            lw = np.r_[lnM, np.full(N, np.log(K))]
            d2 = ((C[:, None, :] - V[None]) ** 2).sum(-1)
            Lg = lw[None] - d2 / (2 * T)
            P0 = np.exp(Lg[:, 0] - logsumexp(Lg, axis=1))
            nfail += (P0 > 0.5).sum(); ntot += len(C)
        row[N] = dict(formula=qf, info_spectrum=qis, gibbs_sim=float(nfail / ntot))
    rep[key] = row
    print(key, {N: {k: round(v, 3) for k, v in r.items()} for N, r in row.items()}, flush=True)
json.dump(rep, open(f'{out}/t2.json', 'w'), indent=1)
