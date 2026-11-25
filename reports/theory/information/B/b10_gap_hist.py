"""b10: (1) generalisation gap (new - trained) vs N; (2) predicted per-voice histogram (Gaussian Lambda + utterance noise) vs observed.
Usage: python3 -I b10_gap_hist.py out/b0.npz"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import norm, linregress
from bt_lib import load
S = load(sys.argv[1]); rng = np.random.default_rng(11)
print("(1) gap q_new - q_trained per N (all K; new = the 4 grid voices; externals when available)")
rows = []
for r in sorted(S, key=lambda r: (S[r]["N"], S[r]["K"], r)):
    if S[r]["var"] not in ("", "wide") or not S[r]["refs"] or S[r]["N"] < 8: continue
    q = {}
    for grp in ["seen", "unseen", "external"]:
        R = [x for x in S[r]["refs"] if x["grp"] == grp]
        if R: q[grp] = 1 - np.mean([x["r1"]["k"] / x["r1"]["n"] for x in R])
    rows.append((S[r]["N"], S[r]["K"], q["unseen"] - q["seen"], q.get("external", np.nan) - q["seen"]))
rows = np.array(rows)
for N in [8, 32, 128, 200]:
    m = rows[:, 0] == N; print(f"  N={N:3d}: gap (4 new) {rows[m,2].mean():+.3f}  gap (externals) {np.nanmean(rows[m,3]) if np.isfinite(rows[m,3]).any() else float('nan'):+.3f}  (n runs {m.sum()})")
m = rows[:, 2] > 0
lr = linregress(np.log(rows[m, 0]), np.log(rows[m, 2])); print(f"  gap(4 new) ~ N^-b : b = {-lr.slope:.2f} +- {lr.stderr:.2f}")
print("\n(2) histogram of the 38 externals (bins <=.1 | .1-.3 | .3-.5 | .5-.7 | .7-.9 | >=.9), Gaussian state Lambda ~ N(mu, sL^2), utterance tau")
OBS = {8: [10, 4, 3, 1, 5, 15], 32: [4, 3, 4, 5, 2, 20], 128: [4, 1, 3, 2, 4, 24]}
Q = {8: .434, 32: .302, 128: .211}
edges = [-.01, .1, .3, .5, .7, .9, 1.01]
def hist(p):
    h = np.zeros(6)
    for v in p:
        if v <= .1: h[0] += 1
        elif v >= .9: h[5] += 1
        else: h[1 + min(int((v - .1) / .2), 3)] += 1
    return h
for lab, ratio in [("ICC .88 (tau/sL=.37)", 0.37), ("Lemma 1, kappa=1 (tau/sL=.085)", 0.085), ("sampling (tau/sL=.76)", 0.76)]:
    s = f"  {lab:32s}:"
    for N in [8, 32, 128]:
        sL = 1.0; tau = ratio; mu = -norm.ppf(Q[N]) * np.sqrt(sL ** 2 + tau ** 2)
        H = np.zeros(6)
        for _ in range(500):
            L = mu + sL * rng.standard_normal(38); p = rng.binomial(25, norm.cdf(L / tau)) / 25; H += hist(p)
        H /= 500; s += f" N={N}: " + "|".join(f"{h:4.1f}" for h in H) + f" (chi2 {np.sum((np.array(OBS[N])-H)**2/np.maximum(H,.5)):.1f})"
    print(s)
print("  observed                        :" + "".join(f" N={N}: " + "|".join(f"{h:4d}" for h in OBS[N]) for N in [8, 32, 128]))
