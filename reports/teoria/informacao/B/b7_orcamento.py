"""b7: (a) orcamento de informacao por voz externa (unidades de log-contagem, s^2=12.1 calibrado nas treinadas);
(b) previsao do tudo-ou-nada por duracao (grandes desvios); (c) WER no gargalo de informacao (dois termos).
Uso: python3 -I b7_orcamento.py out/b0.npz"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import norm, spearmanr, linregress
from scipy.special import logsumexp
from bt_lib import load, Geo
S = load(sys.argv[1]); LN2 = np.log(2); S2 = 12.1; TAU = 0.83; rng = np.random.default_rng(9)
WIDE = ["n8_k32_s1-wide", "n32_k32_s1-wide", "n128_k32_s1-wide"]
def wgeo(g):
    Wreg = g.W + 0.05 * np.trace(g.W) / 32 * np.eye(32); Li = np.linalg.inv(np.linalg.cholesky(Wreg))
    return Li, g.means @ Li.T, Li @ g.mlj, np.trace(Li @ g.W @ Li.T)
print("(a) orcamento por voz externa (bits): Lambda = log2(K/M) + E - C ;  E = (||c-v0||^2 - tr W)/(2 s^2 ln2) ; C = -log2 sum_i exp(-(||c-v_i||^2 - trW - trW/n_i)/(2 s^2))")
out = {}
for r in WIDE:
    g = Geo(S[r]); Li, Mw, v0w, trW = wgeo(g)
    R = [x for x in g.refs if x["grp"] == "external"]; Cw = np.array([x["c_raw"] for x in R]) @ Li.T
    p = np.array([x["r1"]["k"] / x["r1"]["n"] for x in R])
    d0 = ((Cw - v0w) ** 2).sum(1) - trW
    d2 = ((Cw[:, None] - Mw[None]) ** 2).sum(-1) - trW - trW / g.n[None, :]
    E = d0 / (2 * S2 * LN2); C = -logsumexp(-d2 / (2 * S2), axis=1) / LN2
    Cnn = d2.min(1) / (2 * S2 * LN2); neff = 2 ** (Cnn - C)
    prior = np.log2(g.K / 2000); L = prior + E - C
    o = dict(E_med=float(np.median(E)), E_sd=float(E.std()), C_med=float(np.median(C)), Cnn_med=float(np.median(Cnn)), log2neff_med=float(np.median(np.log2(neff))),
             prior=float(prior), L_med=float(np.median(L)), frac_L_pos=float(np.mean(L > 0)), q_obs=float(1 - p.mean()),
             q_pred_threshold=float(np.mean(L <= 0)), q_pred_tau=float(1 - norm.cdf(L * LN2 / TAU).mean()),
             rho_L=float(spearmanr(L, p).correlation), rho_E=float(spearmanr(E, p).correlation), rho_C=float(spearmanr(-C, p).correlation))
    out[r] = o
    print(f"  {r}: E mediana {o['E_med']:.1f} (DP {o['E_sd']:.1f}) | C mediana {o['C_med']:.1f} (vizinho {o['Cnn_med']:.1f}, log2 n_eff {o['log2neff_med']:.1f}) | prior {prior:.2f} "
          f"| Lambda mediana {o['L_med']:+.1f} bits | q prev (limiar) {o['q_pred_threshold']:.2f} (tau) {o['q_pred_tau']:.2f} vs obs {o['q_obs']:.2f} | Spearman copia~Lambda {o['rho_L']:.2f}, ~E {o['rho_E']:.2f}, ~-C {o['rho_C']:.2f}")
# (b) duration: predicted fraction intermediate for short vs long halves from per-voice latent and tau ~ T^-0.376
print("\n(b) tudo-ou-nada x duracao: previsao com tau ~ duracao^-0.376 (b3) a partir do estado por voz")
for r in WIDE:
    fs_o, fl_o, fs_p, fl_p = [], [], [], []
    for x in S[r]["refs"]:
        if x["grp"] != "external": continue
        du = np.array(x["r1"]["per_dur"]); cp = np.array(x["r1"]["per_copy"]); o = np.argsort(du)
        Ts, Tl = du[o[:12]].mean(), du[o[13:]].mean(); T = du.mean()
        Lv = TAU * norm.ppf((cp.sum() + .5) / 26)
        ts, tl = TAU * (Ts / T) ** -0.376, TAU * (Tl / T) ** -0.376
        fs_o.append(0 < cp[o[:12]].mean() < 1); fl_o.append(0 < cp[o[13:]].mean() < 1)
        ps = rng.binomial(12, norm.cdf(Lv / ts), 2000) / 12; pl = rng.binomial(12, norm.cdf(Lv / tl), 2000) / 12
        fs_p.append(np.mean((ps > 0) & (ps < 1))); fl_p.append(np.mean((pl > 0) & (pl < 1)))
    print(f"  {r}: fracao com 0<p<1 — curtas obs {np.mean(fs_o):.2f} prev {np.mean(fs_p):.2f} | longas obs {np.mean(fl_o):.2f} prev {np.mean(fl_p):.2f}")
# (c) WER: two-term information-bottleneck model
print("\n(c) WER = w0 + a_v * sim_default(saida) [termo de voz] + a_n * energia de incomodo")
X, Y = [], []
for r in S:
    if S[r]["var"] not in ("", "wide") or not S[r]["refs"]: continue
    for x in S[r]["refs"]:
        X.append(x["r1"]["sim_def"]); Y.append(x["r1"]["wer"])
X, Y = np.array(X), np.array(Y); lr = linregress(X, Y)
print(f"  transversal (forca 1, R=1, todas as vozes e treinos, n={len(X)}): WER = {lr.intercept:.3f} + {lr.slope:.3f} * sim_default ; r = {lr.rvalue:.2f}")
# within-run version (demeaned by run)
Xd, Yd = [], []
for r in S:
    if S[r]["var"] not in ("", "wide") or not S[r]["refs"]: continue
    xs = np.array([x["r1"]["sim_def"] for x in S[r]["refs"]]); ys = np.array([x["r1"]["wer"] for x in S[r]["refs"]])
    Xd += list(xs - xs.mean()); Yd += list(ys - ys.mean())
lrw = linregress(Xd, Yd); print(f"  dentro do treino: inclinacao {lrw.slope:.3f} (ep {lrw.stderr:.3f}), r = {lrw.rvalue:.2f}")
b_v = lrw.slope
# R=4: voice term from d sim_default, residual -> nuisance coefficient
dW, dS = [], []
for r in WIDE:
    for x in S[r]["refs"]:
        if x["grp"] == "external" and "r4" in x: dW.append(x["r4"]["wer"] - x["r1"]["wer"]); dS.append(x["r4"]["sim_def"] - x["r1"]["sim_def"])
dW, dS = np.array(dW), np.array(dS)
resid = dW.mean() - b_v * dS.mean(); a_n = resid / (8 - 32)
print(f"  R=4 (externas): dWER obs {dW.mean():+.4f}; parte de voz prevista {b_v*dS.mean():+.4f}; resto {resid:+.4f} => a_n = {a_n:.5f} por unidade de ||xi||^2 (32 -> 8)")
for gname in ["seen", "unseen", "external"]:
    w1, w6, s1, s6 = [], [], [], []
    for r in S:
        if S[r]["var"] != "" or not S[r]["refs"]: continue
        for x in S[r]["refs"]:
            if x["grp"] == gname and "s06" in x: w1.append(x["r1"]["wer"]); w6.append(x["s06"]["wer"]); s1.append(x["r1"]["sim_def"]); s6.append(x["s06"]["sim_def"])
    w1, w6, s1, s6 = map(np.array, (w1, w6, s1, s6))
    pv = b_v * (s6 - s1).mean(); pn = a_n * (0.36 * 32 - 32)
    print(f"  forca 0.6 {gname:8s}: dWER obs {np.mean(w6-w1):+.3f} | prev voz {pv:+.3f} + incomodo {pn:+.3f} = {pv+pn:+.3f}  (d sim_default {np.mean(s6-s1):+.3f})")
json.dump(out, open("out/b7.json", "w"), indent=1)
