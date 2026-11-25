"""Tarefa 5: colapso dos dados em x = N^a K^b, para vozes novas (unseen) e treinadas (seen).
(i) potencia q = c N^-a K^-b (q = fracao nao copiada); (ii) probit: Phi^-1(1-q) = c0 + a lnN + b lnK; (iii) margem media = c0 + a lnN + b lnK;
(iv) colapso nao parametrico: direcao theta em (lnN, lnK) que minimiza o SSE de uma regressao isotonica (crescente) de y em x; b/a = tan(theta).
Intervalos: bootstrap de frases (locutores fixos) + ruido de semente N(0, sd_semente^2) por treino; tambem com reamostragem de locutores.
Uso: python3 -I dados_05_colapso.py <results_dir> <out_dir> [B]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from dados_comum import load, main_cond
pd.set_option("display.width", 250)
df = load(sys.argv[1]); out = sys.argv[2]; B = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
d = main_cond(df); rng = np.random.default_rng(55)
seed = json.load(open(os.path.join(out, "seed_sd.json")))
def cube(dd, col): return dd.pivot(index="reference", columns="sent", values=col).sort_index().values
plain = d[d.variant == "plain"]
TH = np.deg2rad(np.arange(-89.5, 90, 0.5))
def estimates(N, K, q, mu):
    lN, lK = np.log(N), np.log(K); X = np.c_[np.ones_like(lN), lN, lK]
    r = {}
    # (i) potencia
    best = (np.inf, None)
    for a in np.arange(-0.5, 1.5001, 0.01):
        x = N[None, :] ** -a * K[None, :] ** -np.arange(-0.5, 2.0001, 0.01)[:, None]
        c = (x * q).sum(1) / (x * x).sum(1); s = ((c[:, None] * x - q) ** 2).sum(1); i = int(np.argmin(s))
        if s[i] < best[0]: best = (s[i], (a, np.arange(-0.5, 2.0001, 0.01)[i]))
    r["pot_a"], r["pot_b"] = best[1]
    z = norm.ppf(np.clip(1 - q, 0.01, 0.99)); bz = np.linalg.lstsq(X, z, rcond=None)[0]; r["probit_a"], r["probit_b"] = bz[1], bz[2]
    bm = np.linalg.lstsq(X, mu, rcond=None)[0]; r["marg_a"], r["marg_b"] = bm[1], bm[2]
    for nm, y in [("iso_q", 1 - q), ("iso_m", mu)]:
        sse = []
        for t in TH:
            x = np.cos(t) * lN + np.sin(t) * lK
            f = IsotonicRegression(increasing=True).fit(x, y); sse.append(((f.predict(x) - y) ** 2).sum())
        sse = np.array(sse); t = TH[int(np.argmin(sse))]
        r[nm + "_theta"] = np.rad2deg(t); r[nm + "_ratio"] = np.tan(t)
    return r
rows = []
for grp in ["unseen", "seen"]:
    for excl2 in ([False, True] if grp == "unseen" else [True]):
        runs = sorted(plain.run.unique()); runs = [x for x in runs if not (excl2 and x.startswith("n2_"))]
        meta = plain.groupby("run")[["N", "K"]].first().loc[runs]; N = meta.N.values.astype(float); K = meta.K.values.astype(float)
        Yc = np.stack([cube(d[(d.run == r) & (d.group == grp)], "copied") for r in runs]); Ym = np.stack([cube(d[(d.run == r) & (d.group == grp)], "m") for r in runs])
        q = 1 - Yc.mean((1, 2)); mu = Ym.mean((1, 2))
        e0 = estimates(N, K, q, mu)
        sq, sm = seed[f"{grp}|copied"]["sd_seed_total"], seed[f"{grp}|m"]["sd_seed_total"]
        for mode in ["frases+semente", "locutores+frases+semente"]:
            bs = []
            for i in range(B):
                a, b = Yc.shape[1], Yc.shape[2]
                si = rng.integers(0, a, a) if mode.startswith("loc") else np.arange(a); ti = rng.integers(0, b, b)
                e = rng.normal(0, 1, len(runs))
                bs.append(estimates(N, K, np.clip(1 - Yc[:, si][:, :, ti].mean((1, 2)) - e * sq, 0, 1), Ym[:, si][:, :, ti].mean((1, 2)) + e * sm))
            bs = pd.DataFrame(bs)
            for k, v in e0.items():
                rows.append(dict(grupo=grp, sem_N2=excl2, modo=mode, est=k, valor=v, lo=bs[k].quantile(.025), hi=bs[k].quantile(.975)))
        # razao b/a e angulo para os parametricos
R = pd.DataFrame(rows); R.to_csv(os.path.join(out, "colapso.csv"), index=False)
print(R.round(3).to_string())
