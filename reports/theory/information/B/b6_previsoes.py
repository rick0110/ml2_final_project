"""b6: per-voice predictions from the measured state (strength 1, R=1) + shift DeltaLambda computed from the geometry.
Scale: 1 nat = Delta ln K (count law). sigma_tot = 1/0.42 (q_seen vs ln K, 04_dados); tau (utterance noise in nats) from the latent ICC;
s0^2 (width of the evidence against the default voice) from b5 (spread of ||v-v0||^2 / sigma_L). Nothing here uses the externals' q(N) curve,
except the item 'large N (fitted gamma)', marked as such.
Usage: python3 -I b6_previsoes.py out/b0.npz [s0sq] [icc]"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import norm, spearmanr
from scipy.special import logsumexp
from bt_lib import load, Geo
S = load(sys.argv[1]); S0 = float(sys.argv[2]) if len(sys.argv) > 2 else 12.1; ICC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.88
SD_D2 = 54.1                                  # b5: SD of ||v-v0||^2 among trained voices (whitened, corrected)
SIGL = SD_D2 / (2 * S0)                       # between-voice SD of the latent state Lambda (nats): quadratic evidence
TAU = SIGL * np.sqrt((1 - ICC) / ICC)         # utterance noise (nats), from the measured latent ICC
SIG = np.sqrt(SIGL ** 2 + TAU ** 2)
KAPPA = 0.42 * SIG                            # dLambda/dlnK of the trained voices implied by the 0.42 probit of q_seen on ln K
TAU_LEMA = 4 * 0.036 / 0.75                   # utterance noise predicted by Lemma 1 + posterior mean (nats)
print(f"kappa (prior weight ~ K^kappa) = {KAPPA:.2f};  tau from ICC = {TAU:.2f} vs tau from Lemma 1 = {TAU_LEMA:.2f} nats (||Delta||^2=0.75; 0.48 if 0.3)")
rng = np.random.default_rng(5)
print(f"s0^2={S0}  sigma_tot={SIG:.2f}  sigma_L={SIGL:.2f}  tau={TAU:.2f} nats")

def wgeo(g):
    Wreg = g.W + 0.05 * np.trace(g.W) / 32 * np.eye(32); Li = np.linalg.inv(np.linalg.cholesky(Wreg))
    return Li, g.means @ Li.T, Li @ g.mlj

def lam_nw(Cw, Mw, v0w, K, s2, logM=np.log(2000), P=None):
    if P is not None: Cw, Mw, v0w = Cw @ P.T, Mw @ P.T, v0w @ P.T
    d2 = ((Cw[:, None, :] - Mw[None, :, :]) ** 2).sum(-1); d0 = ((Cw - v0w) ** 2).sum(1)
    return np.log(K) + logsumexp(-d2 / (2 * s2), axis=1) - logM + d0 / (2 * s2)

def latent_from_rate(k, n):
    return TAU * norm.ppf((k + 0.5) / (n + 1))

def predict(Lv, dL):
    return norm.cdf((Lv + dL) / TAU)

res = {}
# ---------- (A) strength 0.6
print("\n(A) strength 0.6: DeltaLambda = Lambda_NW(0.6 c) - Lambda_NW(c)")
agg = {}
for r in sorted(S):
    if S[r]["var"] not in ("",) or not S[r]["refs"]: continue
    g = Geo(S[r]); Li, Mw, v0w = wgeo(g)
    for j, x in enumerate(g.refs):
        if "s06" not in x: continue
        c = np.array(x["c_raw"]); C2 = np.vstack([c, 0.6 * c]) @ Li.T
        L = lam_nw(C2, Mw, v0w, g.K, S0); dL = L[1] - L[0]
        Lv = latent_from_rate(x["r1"]["k"], x["r1"]["n"])
        a = agg.setdefault(x["grp"], dict(p1=[], p6=[], pr=[], dL=[]))
        a["p1"].append(x["r1"]["k"] / x["r1"]["n"]); a["p6"].append(x["s06"]["k"] / x["s06"]["n"]); a["pr"].append(predict(Lv, dL)); a["dL"].append(dL)
for gname, a in agg.items():
    a = {k: np.array(v) for k, v in a.items()}
    print(f"  {gname:8s} n={len(a['p1'])}: copy 1.0 {a['p1'].mean():.3f} | 0.6 obs {a['p6'].mean():.3f} pred {a['pr'].mean():.3f} | DeltaL median {np.median(a['dL']):+.2f} nats "
          f"| rho(pred,obs) per voice {spearmanr(a['pr'], a['p6']).correlation:.2f}")
    res[f"s06_{gname}"] = dict(obs1=float(a["p1"].mean()), obs6=float(a["p6"].mean()), pred6=float(a["pr"].mean()), dL_med=float(np.median(a["dL"])))
# where is the origin?
g = Geo(S["n128_k32_s1-wide"]); Li, Mw, v0w = wgeo(g)
print("  (n128-wide geometry, whitened) |v0|=%.1f |m_pop|=%.1f |v0-m_pop|=%.1f" % (np.linalg.norm(v0w), np.linalg.norm(Mw.mean(0)), np.linalg.norm(v0w - Mw.mean(0))))

# ---------- (B) R=4: per-voice simulation with projected reference noise
print("\n(B) R=4 (externals, sim_spk criterion): sigma_ref = |u - v0|/s0^2, the noise of the mean includes the original reference")
for r in ["n8_k32_s1-wide", "n32_k32_s1-wide", "n128_k32_s1-wide"]:
    g = Geo(S[r]); Li, Mw, v0w = wgeo(g)
    R = [x for x in g.refs if x["grp"] == "external" and "r4" in x]
    Cw = np.array([x["c_raw"] for x in R]) @ Li.T
    # gradient of Lambda_NW in c (a numerical one would match): (ubar - v0)/s0^2, ubar = posterior-weighted mean of the trained voices
    d2 = ((Cw[:, None, :] - Mw[None]) ** 2).sum(-1); w = np.exp(-(d2 - d2.min(1, keepdims=True)) / (2 * S0)); w /= w.sum(1, keepdims=True)
    ub = w @ Mw; sref = np.linalg.norm(ub - v0w, axis=1) / S0
    p1o = np.array([x["r1"]["k_spk"] / 25 for x in R]); p4o = np.array([x["r4"]["k_spk"] / 25 for x in R])
    sims = []
    q1 = np.clip(p1o.mean(), .02, .98)
    sdb = np.sqrt(max(SIGL ** 2 - np.median(sref) ** 2, 1e-6))
    mu = norm.ppf(q1) * np.sqrt(SIGL ** 2 + TAU ** 2)              # mean of the latent state that reproduces the mean R=1 copy rate
    for _ in range(400):
        Lb = mu + sdb * rng.standard_normal(len(R)); e = rng.standard_normal((len(R), 4)) * sref[:, None]
        L1 = Lb + e[:, 0]; L4 = Lb + e.mean(1)
        p1 = rng.binomial(25, norm.cdf(L1 / TAU)) / 25; p4 = rng.binomial(25, norm.cdf(L4 / TAU)) / 25
        sims.append([np.abs(p4 - p1).mean(), np.mean((p1 > .5) != (p4 > .5)), np.corrcoef(p1, p4)[0, 1], (p4 - p1).mean()])
    sm = np.mean(sims, 0)
    print(f"  {r}: sigma_ref median {np.median(sref):.2f} nats | pred: |dp| {sm[0]:.3f} flips {sm[1]:.3f} corr {sm[2]:.2f} dmean {sm[3]:+.3f} | "
          f"obs: |dp| {np.abs(p4o-p1o).mean():.3f} flips {np.mean((p1o>.5)!=(p4o>.5)):.3f} corr {np.corrcoef(p1o,p4o)[0,1]:.2f} dmean {(p4o-p1o).mean():+.3f}")
    Vb = SIGL ** 2 - np.median(sref) ** 2
    I1 = 0.5 * np.log2(1 + Vb / np.median(sref) ** 2); I4 = 0.5 * np.log2(1 + Vb / (np.median(sref) ** 2 / 4))
    print(f"     information about the decision statistic: R=1 {I1:.2f} bits, R=4 {I4:.2f} bits (gain {I4-I1:+.2f}); 03: gain in z about the voice +2.5/+4.6/+5.0 bits")
    res[f"R4_{r}"] = dict(sref=float(np.median(sref)), pred=sm.tolist(), I1=I1, I4=I4)

# ---------- (C) duration: short vs long utterances within the voice
print("\n(C) all-or-none vs duration (short half x long half of each voice's 25 utterances; wide externals)")
for r in ["n8_k32_s1-wide", "n32_k32_s1-wide", "n128_k32_s1-wide"]:
    ps, pl = [], []
    for x in S[r]["refs"]:
        if x["grp"] != "external": continue
        du = np.array(x["r1"]["per_dur"]); cp = np.array(x["r1"]["per_copy"]); o = np.argsort(du)
        ps.append(cp[o[:12]].mean()); pl.append(cp[o[13:]].mean())
    ps, pl = np.array(ps), np.array(pl)
    ext = lambda p: np.mean(np.abs(p - .5))
    print(f"  {r}: mean |p-1/2| short {ext(ps):.3f} long {ext(pl):.3f}; intermediate fraction (0<p<1) short {np.mean((ps>0)&(ps<1)):.2f} long {np.mean((pl>0)&(pl<1)):.2f}")

# ---------- (D) halved M, large N, latent dimension: shifts applied to the state measured in n128-wide
print("\n(D) predictions from the state measured on the 38 externals (n128_k32_s1-wide, sim_ref criterion)")
g = Geo(S["n128_k32_s1-wide"]); Li, Mw, v0w = wgeo(g)
R = [x for x in g.refs if x["grp"] == "external"]; Lv = np.array([latent_from_rate(x["r1"]["k"], x["r1"]["n"]) for x in R])
q0 = 1 - np.mean([x["r1"]["k"] / x["r1"]["n"] for x in R]); print(f"  observed q (N=128,K=32,M=2000): {q0:.3f}; q from the latent state (consistency) {1-predict(Lv,0).mean():.3f}")
for name, dL in [("M/2, count prior M=2000 (dL=kappa ln2)", KAPPA*np.log(2)), ("M/2, pre-training prior M_eff=15100 (dL=kappa ln(15100/14100))", KAPPA*np.log(15100 / 14100)),
                 ("K=64 externals (dL=0.29 kappa ln2, weight of K on the grid's new voices)", 0.29*KAPPA*np.log(2)), ("M x 2, count", -KAPPA*np.log(2))]:
    print(f"  {name}: q = {1-predict(Lv, dL).mean():.3f}")
GAM = 0.23 * SIG   # = gamma nats per ln N
for N in [512, 2048]:
    print(f"  N={N} (gamma={GAM:.2f} nats per ln N, FITTED on the external curve): q = {1-predict(Lv, GAM*np.log(N/128)).mean():.3f};  with gamma=1 (LLN count): {1-predict(Lv, np.log(N/128)).mean():.3f}")
# dimension: top-k PCs of whitened total covariance of training codes
lat = S["n128_k32_s1-wide"]; 
Ctr = np.vstack([Mw, v0w[None]]); wts = np.r_[np.full(len(Mw), g.K), 2000.0]
mu_ = (wts[:, None] * Ctr).sum(0) / wts.sum(); Sig = ((Ctr - mu_).T * wts) @ (Ctr - mu_) / wts.sum() + np.eye(32)
ev, V = np.linalg.eigh(Sig); V = V[:, ::-1]
Cw = np.array([x["c_raw"] for x in R]) @ Li.T
L32 = lam_nw(Cw, Mw, v0w, g.K, S0)
for k in [1, 2, 4, 8, 16]:
    P = V[:, :k].T; Lk = lam_nw(Cw, Mw, v0w, g.K, S0, P=P)
    dL = Lk - L32
    print(f"  latent projected onto k={k:2d} principal directions: q = {1-predict(Lv, dL).mean():.3f}  (dL median {np.median(dL):+.2f}, NW decision unchanged for {np.mean(np.sign(Lk)==np.sign(L32)):.2f} of the voices)")
json.dump(res, open("out/b6.json", "w"), indent=1, default=float)
