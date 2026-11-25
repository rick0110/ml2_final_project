"""b15: (1) per-voice information in the audio vs per-voice copy; (2) minimal 3-parameter model with a calibration/test split;
(3) duration effect with bootstrap and permutation; (4) M/2 with the derived M_eff = 2M+NK.
Usage: python3 -I b15_min.py out/b0.npz out/b12_emb.npz"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import norm, spearmanr
from scipy.special import logsumexp
from scipy.optimize import minimize, brentq
from bt_lib import load, Geo
S = load(sys.argv[1]); d = np.load(sys.argv[2], allow_pickle=False); rng = np.random.default_rng(4)
keys = d["keys"].astype(str); E = d["emb"] / np.linalg.norm(d["emb"], axis=1, keepdims=True); K_ = [k.split("|") for k in keys]
oth = {}
for i, k in enumerate(K_):
    if k[0] == "oth": oth.setdefault(k[1], []).append(E[i])
WIDE = {8: "n8_k32_s1-wide", 32: "n32_k32_s1-wide", 128: "n128_k32_s1-wide"}
print("(1) per-voice information (audio, utterance s0) x per-voice copy")
for N in [8, 128]:
    r = WIDE[N]; R = [x for x in S[r]["refs"] if x["grp"] == "external"]
    ids = [x["ref"] for x in R]; spk = [i.split("_")[0] for i in ids]
    cent = np.array([np.mean(oth[s], 0) for s in spk]); cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    O = np.array([E[[j for j, k in enumerate(K_) if k[0] == "out" and k[1] == r and k[2] == "strength_1.0" and k[3] == i + "__s0"][0]] for i in ids])
    sim = O @ cent.T; b = 30.0
    cv = np.log2(len(ids)) + np.array([(b * sim[i, i] - logsumexp(b * sim[i])) / np.log(2) for i in range(len(ids))])
    p = np.array([x["r1"]["k"] / 25 for x in R])
    print(f"  N={N}: Spearman(c_v, copy) = {spearmanr(cv, p).correlation:.2f}; mean c_v copied(p>.5) {cv[p>.5].mean():.2f} vs not {cv[p<=.5].mean():.2f} bits")
# ---- (2) minimal model
def G(r):  # geometric LRT statistic (b7, s2=12.1), standardized within run
    g = Geo(S[r]); Wr = g.W + 0.05 * np.trace(g.W) / 32 * np.eye(32); Li = np.linalg.inv(np.linalg.cholesky(Wr))
    Mw = g.means @ Li.T; v0 = Li @ g.mlj; trW = np.trace(Li @ g.W @ Li.T)
    idx = [j for j, x in enumerate(g.refs) if x["grp"] == "external"]; Cw = g.C[idx] @ Li.T
    d0 = ((Cw - v0) ** 2).sum(1) - trW; d2 = ((Cw[:, None] - Mw[None]) ** 2).sum(-1) - trW * (1 + 1 / g.n[None, :])
    L = d0 / 24.2 + logsumexp(-d2 / 24.2, axis=1)
    k = np.array([g.refs[j]["r1"]["k"] for j in idx]); return (L - L.mean()) / L.std(), k
DAT = {N: G(WIDE[N]) for N in WIDE}
def probs(N, a, th, gm): z, _ = DAT[N]; return norm.cdf(a * z - th + gm * np.log(N / 128))
def nll_voice(N, a, th, gm):
    z, k = DAT[N]; p = np.clip(probs(N, a, th, gm), 1e-6, 1 - 1e-6); return -(k * np.log(p) + (25 - k) * np.log(1 - p)).sum()
o = minimize(lambda t: nll_voice(128, np.exp(t[0]), t[1], 0), [0, -1], method="Nelder-Mead"); a, th = np.exp(o.x[0]), o.x[1]
gm = brentq(lambda g_: (1 - probs(8, a, th, g_).mean()) - (1 - DAT[8][1].sum() / (25 * len(DAT[8][1]))), -5, 5)
print(f"\n(2) minimal model: p_v = Phi(a*G_v - theta + gamma*ln(N/128)); a={a:.2f}, theta={th:.2f} (per-voice likelihood, N=128 ONLY); gamma={gm:.3f} (q(8) ONLY)")
out = {}
for N in [8, 32, 128]:
    z, k = DAT[N]; p = k / 25; pr = probs(N, a, th, gm); sim = rng.binomial(25, np.tile(pr, (500, 1))) / 25
    ll0 = -np.sum(k * np.log2(p.mean()) + (25 - k) * np.log2(1 - p.mean())) / (25 * len(k)); ll = nll_voice(N, a, th, gm) / np.log(2) / (25 * len(k))
    icc_sim = []
    out[N] = dict(q_obs=float(1 - p.mean()), q_pred=float(1 - pr.mean()), rho=float(spearmanr(pr, p).correlation),
                  mid_obs=float(np.mean((p > .1) & (p < .9))), mid_pred=float(np.mean((sim > .1) & (sim < .9))),
                  bits_model=float(ll), bits_intercept=float(ll0))
    o_ = out[N]; tag = {8: "q fitted (gamma); rest predicted", 32: "EVERYTHING predicted", 128: "a, theta fitted"}[N]
    print(f"  N={N:3d} [{tag}]: q obs {o_['q_obs']:.3f} pred {o_['q_pred']:.3f} | Spearman {o_['rho']:.2f} | voices 10-90% obs {o_['mid_obs']:.2f} pred {o_['mid_pred']:.2f} | loss {o_['bits_model']:.3f} vs intercept-only {o_['bits_intercept']:.3f} bit/sample")
# ---- (3) duration effect: bootstrap + permutation
print("\n(3) duration: statistic T = mean over voices of |p_long - 1/2| - |p_short - 1/2| (12 x 12 utterances); externals, 3 runs")
items = []
for N, r in WIDE.items():
    for x in S[r]["refs"]:
        if x["grp"] != "external": continue
        items.append((x["spk"], np.array(x["r1"]["per_dur"]), np.array(x["r1"]["per_copy"]), np.array(x["r1"]["per_m"])))
def stat(it, perm=False):
    v = []; mids = []
    for _, du, cp, m in it:
        o_ = rng.permutation(25) if perm else np.argsort(du)
        s, l = cp[o_[:12]].mean(), cp[o_[13:]].mean(); v.append(abs(l - .5) - abs(s - .5)); mids.append(((0 < s < 1) * 1.0, (0 < l < 1) * 1.0))
    mids = np.array(mids); return np.mean(v), mids[:, 1].mean() - mids[:, 0].mean()
T0, M0 = stat(items)
spks = sorted(set(i[0] for i in items)); by = {s_: [i for i in items if i[0] == s_] for s_ in spks}
bs = np.array([stat([i for s_ in rng.choice(spks, len(spks)) for i in by[s_]]) for _ in range(2000)])
pm = np.array([stat(items, perm=True) for _ in range(5000)])
print(f"  T = {T0:+.4f} [95% CI speaker bootstrap {np.percentile(bs[:,0],2.5):+.4f}, {np.percentile(bs[:,0],97.5):+.4f}]; permutation p (one-sided) = {np.mean(pm[:,0] >= T0):.4f}")
print(f"  diff. intermediate fraction (long - short) = {M0:+.3f} [95% CI {np.percentile(bs[:,1],2.5):+.3f}, {np.percentile(bs[:,1],97.5):+.3f}]; permutation p = {np.mean(pm[:,1] <= M0):.4f}")
# margin dispersion version: within-voice SD of margin, long vs short halves
ratio = []
for _, du, cp, m in items:
    o_ = np.argsort(du); ratio.append(np.log(m[o_[13:]].std(ddof=1) / m[o_[:12]].std(ddof=1)))
ratio = np.array(ratio); bsr = [np.mean(ratio[rng.integers(0, len(ratio), len(ratio))]) for _ in range(2000)]
print(f"  margin SD ratio long/short = {np.exp(ratio.mean()):.3f} [95% CI {np.exp(np.percentile(bsr,2.5)):.3f}, {np.exp(np.percentile(bsr,97.5)):.3f}] (predicted by the CLT with correlated frames: (T_l/T_s)^-0.38)")
dur_ratio = np.mean([np.log(np.sort(du)[13:].mean() / np.sort(du)[:12].mean()) for _, du, _, _ in items])
print(f"  mean duration ratio long/short = {np.exp(dur_ratio):.2f} => predicted {np.exp(dur_ratio)**-0.38:.3f} (exponent -0.38) / {np.exp(dur_ratio)**-0.5:.3f} (-0.5)")
# ---- (4) M/2 under derived M_eff
z = np.array([norm.ppf((x["r1"]["k"] + .5) / 26) for x in S[WIDE[128]]["refs"] if x["grp"] == "external"])
base = 1 - norm.cdf(z).mean()
for nm, dl in [("M_eff = 2M+NK derived (ln 8096/6096 = 0.28)", np.log(8096 / 6096)), ("naive count (ln 2)", np.log(2))]:
    print(f"  M/2, {nm}: q {base:.3f} -> {1-norm.cdf(z + 1.21*dl).mean():.3f}")
json.dump(dict(min=out, T=T0, Tci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), p_perm=float(np.mean(pm[:, 0] >= T0))), open("out/b15.json", "w"), indent=1, default=float)
