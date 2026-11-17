"""b9: bits usados pelo decodificador (gargalo), WER das treinadas vs K, ICC previsto por ligacao, deslocamento exigido pela forca 0.6,
limiar de vazamento por quadro. Uso: python3 -I b9_extras.py out/b0.npz"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import norm, linregress
from scipy.optimize import brentq
from bt_lib import load, Geo
S = load(sys.argv[1]); TAU = 0.83; rng = np.random.default_rng(3)
r = "n128_k32_s1-wide"; g = Geo(S[r]); Wreg = g.W + 0.05 * np.trace(g.W) / 32 * np.eye(32); Li = np.linalg.inv(np.linalg.cholesky(Wreg))
lam = np.clip(np.linalg.eigvalsh(Li @ g.Sb @ Li.T)[::-1], 0, None)
print("(1) bits de voz: I = 1/2 sum log2(1 + lambda_j / s2) (autovalores branqueados entre locutores, n128-wide)")
for s2, lab in [(1, "ruido de frase so (codificador, mu)"), (12.1, "decodificador, escala de contagem"), (28, "decodificador, escala Lema1/largura logistica"), (36, "")]:
    I = 0.5 * np.log2(1 + lam / s2); I4 = 0.5 * np.log2(1 + lam / (s2 - 1 + 0.25)) if s2 > 1 else 0.5 * np.log2(1 + lam / 0.25)
    cum = np.cumsum(I) / I.sum()
    print(f"  s2={s2:5.1f} {lab:45s}: I={I.sum():5.2f} bits; R=4: {np.sum(I4):5.2f} (ganho {np.sum(I4)-I.sum():+.2f}); fracao em k=1,2,4,8: {cum[0]:.2f} {cum[1]:.2f} {cum[3]:.2f} {cum[7]:.2f}")
print("\n(2) WER das treinadas vs K (treinos plain/wide N>=8): excesso sobre WER das novas em N>=32")
K_, W_ = [], []
for rr in S:
    if S[rr]["var"] not in ("", "wide") or S[rr]["N"] < 8 or not S[rr]["refs"]: continue
    ws = np.mean([x["r1"]["wer"] for x in S[rr]["refs"] if x["grp"] == "seen"]); K_.append(S[rr]["K"]); W_.append(ws)
K_, W_ = np.array(K_), np.array(W_)
for k in [8, 32, 128]: print(f"  K={k}: WER treinadas {W_[K_==k].mean():.3f} (n={np.sum(K_==k)})")
w_inf = 0.07
lr = linregress(np.log(K_), np.log(W_ - w_inf)); print(f"  ajuste (WER - {w_inf}) ~ K^-b: b = {-lr.slope:.2f} +- {lr.stderr:.2f}")
print("\n(3) ICC latente previsto: sigma_L=2.23 (nats-contagem). Ligacao media posterior (tau = 4 sigma_w/||D||^2 / kappa) vs amostragem (tau ~ 1.7)")
for nm, tau in [("media posterior, ||D||^2=0.75, kappa=1", 0.19), ("media posterior, ||D||^2=0.30, kappa=1", 0.48), ("amostragem da posterior (logistica)", 1.7), ("medido (ICC .83-.94)", 0.83)]:
    print(f"  {nm:42s}: tau={tau:.2f}  ICC={2.23**2/(2.23**2+tau**2):.3f}")
print("\n(4) forca 0.6: deslocamento uniforme exigido (unidades de tau) vs previsto pela geometria NW (-1.5 a -2.3 nats ~ -4 a -6 tau)")
for grp in ["seen", "unseen", "external"]:
    L1, p6 = [], []
    for rr in S:
        if S[rr]["var"] != "" or not S[rr]["refs"]: continue
        for x in S[rr]["refs"]:
            if x["grp"] == grp and "s06" in x: L1.append(norm.ppf((x["r1"]["k"] + .5) / 26)); p6.append(x["s06"]["k"] / 25)
    L1 = np.array(L1); target = np.mean(p6)
    sh = brentq(lambda s: norm.cdf(L1 + s).mean() - target, -10, 5)
    print(f"  {grp:8s}: copia 0.6 obs {target:.3f}; deslocamento exigido {sh:+.2f} tau ({sh*TAU:+.2f} nats-contagem)")
print("\n(5) vazamento nas unidades (duas etapas): evidencia acumulada T_eff * eta * D_u contra a margem por voz (estado n128-wide)")
R = [x for x in g.refs if x["grp"] == "external"]; Lv = np.array([TAU * norm.ppf((x["r1"]["k"] + .5) / (x["r1"]["n"] + 1)) for x in R])
print(f"  margem mediana das vozes copiadas: {np.median(Lv[Lv>0]):.2f} nats; q base {1-norm.cdf(Lv/TAU).mean():.3f}")
for Teff in [20, 200]:
    for eta in [1.0, 1 / 25]:
        thr = brentq(lambda D: (1 - norm.cdf((Lv - Teff * eta * D) / TAU).mean()) - 0.30, 0, 50)
        thr2 = brentq(lambda D: (1 - norm.cdf((Lv - Teff * eta * D) / TAU).mean()) - 0.25, 0, 50)
        print(f"  T_eff={Teff:3d} eta={eta:.2f}: q sobe de 0.21 para 0.25 com D_u = {thr2/np.log(2):.3f} bit/quadro; para 0.30 com {thr/np.log(2):.3f} bit/quadro")
