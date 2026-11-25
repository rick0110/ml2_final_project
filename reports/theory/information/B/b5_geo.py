"""b5: geometric quantities measured in the latent (copying not used): LJ-LibriTTS decision axis, ICC along it (=> K0 of the design effect),
spread of d^2 = ||v-v0||^2 among trained voices (=> evidence scale s0^2), reference noise projected onto the gradient.
Usage: python3 -I b5_geo.py out/b0.npz"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bt_lib import load, Geo
S = load(sys.argv[1])
RUNS = sorted([r for r in S if S[r]["var"] in ("", "wide") and S[r]["refs"] and S[r]["N"] >= 32], key=lambda r: (S[r]["N"], S[r]["K"], r))
out = {}
for r in RUNS:
    g = Geo(S[r]); Wreg = g.W + 0.05 * np.trace(g.W) / 32 * np.eye(32); Li = np.linalg.inv(np.linalg.cholesky(Wreg))
    M = g.means @ Li.T; v0 = Li @ g.mlj; Sb = Li @ g.Sb @ Li.T; n = g.n
    mpop = M.mean(0); dvec = v0 - mpop; dist = np.linalg.norm(dvec); u = dvec / dist
    sb_ax = float(u @ Sb @ u); w_ax = float(u @ (Li @ g.W @ Li.T) @ u)
    icc = sb_ax / (sb_ax + w_ax); K0 = w_ax / sb_ax
    # noise-corrected spread of squared distance to LJ among trained voices
    d2 = ((M - v0) ** 2).sum(1) - np.trace(Li @ g.W @ Li.T) / n           # remove E||mean noise||^2
    # sd of d^2 corrected for noise: var(d2_obs) = var(d2_true) + 4 E[(v-v0)' (W/n) (v-v0)] + ...
    noise_var = np.mean(4 * ((M - v0) ** 2).sum(1) / n)
    sd_d2 = float(np.sqrt(max(np.var(d2) - noise_var, 1e-9)))
    out[r] = dict(N=g.N, K=g.K, dist_LJ_pop=float(dist), sb_axis=sb_ax, w_axis=w_ax, icc_axis=icc, K0=K0,
                  mean_d=float(np.sqrt(np.maximum(d2, 0)).mean()), sd_d2=sd_d2, sd_d=float(np.sqrt(np.maximum(d2, 0)).std()),
                  trSb=float(np.trace(Sb)), post_mean=float(np.trace(Li @ np.diag(g.s2lib) @ Li.T) / 32))
    print(r, {k: round(v, 3) for k, v in out[r].items()})
icc = np.array([o["icc_axis"] for o in out.values()]); K0 = np.array([o["K0"] for o in out.values()])
sd = np.array([o["sd_d2"] for o in out.values()]); md = np.array([o["mean_d"] for o in out.values()])
print("\nICC on the decision axis: median %.3f (min %.3f max %.3f) => K0 = (1-ICC)/ICC median %.1f" % (np.median(icc), icc.min(), icc.max(), np.median(K0)))
print("SD of ||v-v0||^2 among trained voices (corrected): median %.1f ; mean distance %.1f" % (np.median(sd), np.median(md)))
SIG_TOT = 1 / 0.42   # 04_dados: probit of q_seen on ln K, b = 0.42 [0.35, 0.53]
for icc_lat in [0.83, 0.88, 0.94]:
    sigL = SIG_TOT * np.sqrt(icc_lat); tau = SIG_TOT * np.sqrt(1 - icc_lat)
    s0 = np.median(sd) / (2 * sigL)
    print(f"latent ICC {icc_lat}: sigma_L={sigL:.2f} nats, tau_w={tau:.2f} nats  => s0^2 = sd(d2)/(2 sigma_L) = {s0:.1f}; "
          f"sigma_ref(R=1) = |v0-u|/s0^2 = {np.median([o['dist_LJ_pop'] for o in out.values()])/s0:.2f} nats")
json.dump(out, open("out/b5.json", "w"), indent=1)
