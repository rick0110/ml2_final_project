"""b16: minimal 3-parameter model (Gaussian per-voice state, ordered by the parameter-free geometric statistic):
 L_v ~ N(mu_N, 1) across voices; per-utterance copy ~ Phi(L_v / t);  mu_N = mu_128 + gamma ln(N/128).
 t and mu_128: marginal likelihood of the per-voice counts at N=128 ONLY; gamma: q(N=8) ONLY. Predicted: all of N=32, the N=8 histograms, ICC.
 The ordering of the voices is predicted by the geometric Lambda statistic (b7/b15), with no parameter.
Usage: python3 -I b16_min3.py out/b0.npz"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import norm, binom
from scipy.optimize import minimize, brentq
from bt_lib import load
S = load(sys.argv[1]); rng = np.random.default_rng(6)
WIDE = {8: "n8_k32_s1-wide", 32: "n32_k32_s1-wide", 128: "n128_k32_s1-wide"}
K = {N: np.array([x["r1"]["k"] for x in S[r]["refs"] if x["grp"] == "external"]) for N, r in WIDE.items()}
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(60); gh_w = gh_w / gh_w.sum()
def pv_grid(mu, t): return np.clip(norm.cdf((mu + gh_x) / t), 1e-9, 1 - 1e-9)
def nll(mu, t, k):
    P = pv_grid(mu, t); from scipy.special import gammaln; lc = gammaln(26) - gammaln(k + 1) - gammaln(26 - k); L = np.exp(lc[:, None] + k[:, None] * np.log(P[None, :]) + (25 - k[:, None]) * np.log(1 - P[None, :])) @ gh_w; return -np.log(np.clip(L, 1e-300, None)).sum()
o = minimize(lambda th: nll(th[0], np.exp(th[1]), K[128]), [1, -1], method="Nelder-Mead"); mu128, t = o.x[0], np.exp(o.x[1])
qN = lambda mu: 1 - (pv_grid(mu, t) @ gh_w)
gam = brentq(lambda g: qN(mu128 + g * np.log(8 / 128)) - (1 - K[8].sum() / (25 * len(K[8]))), -5, 5)
icc = 1 / (1 + t ** 2)
print(f"parameters: mu_128 = {mu128:.3f}, t = {t:.3f} (latent ICC = {icc:.3f}), gamma = {gam:.3f} per ln N")
def hist(p):
    b = np.array([.1, .3, .5, .7, .9]); h = np.zeros(6)
    for v in p: h[0 if v <= .1 else 5 if v >= .9 else 1 + np.searchsorted(b[1:-1], v, side="right")] += 1
    return h
res = dict(mu128=mu128, t=t, gamma=gam, icc=icc, rows={})
for N in [8, 32, 128]:
    mu = mu128 + gam * np.log(N / 128); k = K[N]; p = k / 25
    sims = np.array([hist(rng.binomial(25, norm.cdf((mu + rng.standard_normal(len(k))) / t)) / 25) for _ in range(2000)])
    H = sims.mean(0); Ho = hist(p); chi = np.sum((Ho - H) ** 2 / np.maximum(H, .5))
    pchi = np.mean([np.sum((s - H) ** 2 / np.maximum(H, .5)) >= chi for s in sims])
    midp = (H[1:5].sum()) / len(k); mido = Ho[1:5].sum() / len(k)
    tag = {8: "q fitted (gamma); histogram predicted", 32: "EVERYTHING predicted", 128: "fitted (mu, t)"}[N]
    print(f"  N={N:3d} [{tag}]: q obs {1-p.mean():.3f} pred {qN(mu):.3f} | voices 10-90% obs {mido:.2f} pred {midp:.2f} | hist obs {Ho.astype(int).tolist()} pred {np.round(H,1).tolist()} chi2 {chi:.1f} (p sim {pchi:.2f})")
    res["rows"][N] = dict(q_obs=float(1 - p.mean()), q_pred=float(qN(mu)), mid_obs=float(mido), mid_pred=float(midp), chi2=float(chi), p=float(pchi))
for N in [512, 2048]:
    print(f"  extrapolation N={N}: q = {qN(mu128 + gam*np.log(N/128)):.3f}")
json.dump(res, open("out/b16.json", "w"), indent=1, default=float)
