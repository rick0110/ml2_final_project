"""b17: which mechanism produces the ~1/22 factor? Test via strength 0.6.
 fixed decoder noise (capacity): I(a) = 1/2 sum log2(1 + a^2 lambda/s2)  -> falls with a
 posterior/KL noise (before the scaling): scales signal and noise together -> I does not change
 default-voice prior: changes which voices collapse, not the resolution among the non-collapsed ones
Also: bits in mu vs z (cost of the KL). Usage: python3 -I b17_forca.py out/b0.npz out/b13.json"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import brentq
from scipy.linalg import eigh
from bt_lib import load, Geo
S = load(sys.argv[1]); R = json.load(open(sys.argv[2]))
g = Geo(S["n128_k32_s1-wide"]); Wr = g.W + 0.05 * np.trace(g.W) / 32 * np.eye(32); Li = np.linalg.inv(np.linalg.cholesky(Wr))
Sb = Li @ g.Sb @ Li.T; P = Li @ np.diag(g.s2lib) @ Li.T; I_ = Li @ g.W @ Li.T
lam = np.clip(np.linalg.eigvalsh(Sb), 0, None)
Imu = 0.5 * np.log2(1 + np.clip(eigh(Sb, I_, eigvals_only=True), 0, None)).sum()
Iz = 0.5 * np.log2(1 + np.clip(eigh(Sb, I_ + P, eigvals_only=True), 0, None)).sum()
print(f"voice bits in the latent: mu {Imu:.1f}, z = mu + posterior noise (KL) {Iz:.1f}  => cost of the KL {Imu-Iz:.1f} bits")
f = lambda s2, a=1.0: 0.5 * np.log2(1 + a * a * lam / s2).sum()
for run in ["n128_k8_s1", "n128_k32_s1", "n200_k32_s1"]:
    for est in ["ce", "fano"]:
        c1 = R[f"plain|{run}|strength_1.0"][est]; c6 = R[f"plain|{run}|strength_0.6"][est]
        s2 = brentq(lambda s: f(s) - c1, 1e-4, 1e7)
        print(f"  {run:12s} [{est:4s}] C(1.0) {c1:.2f}  C(0.6) obs {c6:.2f} | pred decoder-noise {f(s2, .6):.2f} | pred KL/posterior {c1:.2f}")
