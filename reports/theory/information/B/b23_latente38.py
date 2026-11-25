"""b23: latent link for 38 voices (same estimator: identification + Fano), with the covariances measured in each run:
voice v ~ N(0, Sigma_b); one reference utterance: mu = v + w, w ~ N(0, W); z = mu + posterior noise. Centroids = true voices.
Usage: python3 -I b23_latente38.py out/b0.npz"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bt_lib import load, Geo
S = load(sys.argv[1]); rng = np.random.default_rng(5)
def fano(acc, n):
    pe = min(max(1 - acc, 1e-12), 1 - 1e-12); return np.log2(n) - (-(pe*np.log2(pe)+(1-pe)*np.log2(1-pe))) - pe*np.log2(n-1)
for run in ["n8_k32_s1-wide", "n32_k32_s1-wide", "n128_k32_s1-wide"]:
    g = Geo(S[run]); Wr = g.W + 0.05 * np.trace(g.W) / 32 * np.eye(32); Li = np.linalg.inv(np.linalg.cholesky(Wr))
    Sb = Li @ g.Sb @ Li.T; ev, V = np.linalg.eigh(Sb); Hb = V * np.sqrt(np.clip(ev, 0, None)); Wn = Li @ g.W @ Li.T; P = Li @ np.diag(g.s2lib) @ Li.T
    out = []
    for kind, Nn in [("mu", Wn), ("z", Wn + P)]:
        Ln = np.linalg.cholesky(Nn + 1e-9 * np.eye(32)); Lw = np.linalg.inv(np.linalg.cholesky(Nn + 1e-9 * np.eye(32)))
        acc = []
        for _ in range(300):
            v = rng.standard_normal((38, 32)) @ Hb.T; x = v + rng.standard_normal((38, 32)) @ Ln.T
            d = ((((x[:, None] - v[None]) @ Lw.T)) ** 2).sum(-1); acc.append(np.mean(d.argmin(1) == np.arange(38)))
        out.append(f"{kind}: accuracy {np.mean(acc):.3f} Fano {fano(np.mean(acc), 38):.2f}")
    print(f"{run}: " + " | ".join(out))
