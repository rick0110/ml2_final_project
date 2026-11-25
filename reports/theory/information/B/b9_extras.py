"""b9: bits used by the decoder (bottleneck), WER of the trained voices vs K, ICC predicted by each link, shift required by strength 0.6,
per-frame leakage threshold. Usage: python3 -I b9_extras.py out/b0.npz"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import norm, linregress
from scipy.optimize import brentq
from bt_lib import load, Geo
S = load(sys.argv[1]); TAU = 0.83; rng = np.random.default_rng(3)
r = "n128_k32_s1-wide"; g = Geo(S[r]); Wreg = g.W + 0.05 * np.trace(g.W) / 32 * np.eye(32); Li = np.linalg.inv(np.linalg.cholesky(Wreg))
lam = np.clip(np.linalg.eigvalsh(Li @ g.Sb @ Li.T)[::-1], 0, None)
print("(1) voice bits: I = 1/2 sum log2(1 + lambda_j / s2) (whitened between-speaker eigenvalues, n128-wide)")
for s2, lab in [(1, "utterance noise only (encoder, mu)"), (12.1, "decoder, count scale"), (28, "decoder, Lemma1/logistic-width scale"), (36, "")]:
    I = 0.5 * np.log2(1 + lam / s2); I4 = 0.5 * np.log2(1 + lam / (s2 - 1 + 0.25)) if s2 > 1 else 0.5 * np.log2(1 + lam / 0.25)
    cum = np.cumsum(I) / I.sum()
    print(f"  s2={s2:5.1f} {lab:45s}: I={I.sum():5.2f} bits; R=4: {np.sum(I4):5.2f} (gain {np.sum(I4)-I.sum():+.2f}); fraction in k=1,2,4,8: {cum[0]:.2f} {cum[1]:.2f} {cum[3]:.2f} {cum[7]:.2f}")
print("\n(2) WER of the trained voices vs K (plain/wide runs N>=8): excess over the WER of the new voices at N>=32")
K_, W_ = [], []
for rr in S:
    if S[rr]["var"] not in ("", "wide") or S[rr]["N"] < 8 or not S[rr]["refs"]: continue
    ws = np.mean([x["r1"]["wer"] for x in S[rr]["refs"] if x["grp"] == "seen"]); K_.append(S[rr]["K"]); W_.append(ws)
K_, W_ = np.array(K_), np.array(W_)
for k in [8, 32, 128]: print(f"  K={k}: WER trained {W_[K_==k].mean():.3f} (n={np.sum(K_==k)})")
w_inf = 0.07
lr = linregress(np.log(K_), np.log(W_ - w_inf)); print(f"  fit (WER - {w_inf}) ~ K^-b: b = {-lr.slope:.2f} +- {lr.stderr:.2f}")
print("\n(3) predicted latent ICC: sigma_L=2.23 (count-nats). Posterior-mean link (tau = 4 sigma_w/||D||^2 / kappa) vs sampling (tau ~ 1.7)")
for nm, tau in [("posterior mean, ||D||^2=0.75, kappa=1", 0.19), ("posterior mean, ||D||^2=0.30, kappa=1", 0.48), ("posterior sampling (logistic)", 1.7), ("measured (ICC .83-.94)", 0.83)]:
    print(f"  {nm:42s}: tau={tau:.2f}  ICC={2.23**2/(2.23**2+tau**2):.3f}")
print("\n(4) strength 0.6: required uniform shift (units of tau) vs predicted by the NW geometry (-1.5 to -2.3 nats ~ -4 to -6 tau)")
for grp in ["seen", "unseen", "external"]:
    L1, p6 = [], []
    for rr in S:
        if S[rr]["var"] != "" or not S[rr]["refs"]: continue
        for x in S[rr]["refs"]:
            if x["grp"] == grp and "s06" in x: L1.append(norm.ppf((x["r1"]["k"] + .5) / 26)); p6.append(x["s06"]["k"] / 25)
    L1 = np.array(L1); target = np.mean(p6)
    sh = brentq(lambda s: norm.cdf(L1 + s).mean() - target, -10, 5)
    print(f"  {grp:8s}: copy 0.6 obs {target:.3f}; required shift {sh:+.2f} tau ({sh*TAU:+.2f} count-nats)")
print("\n(5) leakage into the units (two stages): accumulated evidence T_eff * eta * D_u against the per-voice margin (n128-wide state)")
R = [x for x in g.refs if x["grp"] == "external"]; Lv = np.array([TAU * norm.ppf((x["r1"]["k"] + .5) / (x["r1"]["n"] + 1)) for x in R])
print(f"  median margin of the copied voices: {np.median(Lv[Lv>0]):.2f} nats; base q {1-norm.cdf(Lv/TAU).mean():.3f}")
for Teff in [20, 200]:
    for eta in [1.0, 1 / 25]:
        thr = brentq(lambda D: (1 - norm.cdf((Lv - Teff * eta * D) / TAU).mean()) - 0.30, 0, 50)
        thr2 = brentq(lambda D: (1 - norm.cdf((Lv - Teff * eta * D) / TAU).mean()) - 0.25, 0, 50)
        print(f"  T_eff={Teff:3d} eta={eta:.2f}: q rises from 0.21 to 0.25 with D_u = {thr2/np.log(2):.3f} bit/frame; to 0.30 with {thr/np.log(2):.3f} bit/frame")
