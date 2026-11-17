"""Tarefa 6: artefato da medida vs efeito real nas 38 vozes externas (-wide).
Uso: python3 -I dados_06_artefato.py <results_dir> <out_dir> [B]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from dados_comum import load
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
df = load(sys.argv[1]); out = sys.argv[2]; B = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
rng = np.random.default_rng(66)

# ---------- 0. "proximidade da referencia a voz padrao" c_s = sim_ref quando a saida e a voz padrao ----------
# verdade disponivel so para as 4 vozes novas: treinos -spk a forca 0 (saida = voz padrao, sim_default ~ 0.9)
z = df[(df.strength == 0.0) & (df.variant == "spk") & (df.group == "unseen")]
c_true = z.groupby("spk").sim_ref.mean(); D0 = z.sim_default.mean()
print(f"forca 0 (-spk): sim_default medio = {D0:.3f}; c_s verdadeiro (4 vozes novas) = {c_true.round(3).to_dict()}")
def extrap(x):
    """regressao sim_ref ~ sim_default dentro do locutor, avaliada em sim_default = D0."""
    b = np.polyfit(x.sim_default, x.sim_ref, 1); return np.polyval(b, D0), b[0]
g = df[(df.variant == "plain") & (df.group == "unseen") & (df.R == 1) & (df.strength > 0)]
val = pd.DataFrame({s: extrap(x) for s, x in g.groupby("spk")}, index=["c_hat", "inclinacao"]).T
val["c_true"] = c_true; val["sim_default_max_obs"] = g.groupby("spk").sim_default.max()
print("validacao da extrapolacao (vozes novas, treinos plain forca 0.6 e 1.0):"); print(val.round(3).to_string())

# ---------- 1. dados wide externos ----------
W = df[(df.variant == "wide") & (df.group == "external") & (df.strength == 1.0) & (df.R == 1)].copy()
cs_ex = pd.Series({s: extrap(x)[0] for s, x in W.groupby("spk")}, name="c_extrap")
# a extrapolacao falha na validacao (ver acima); usamos como proxy o nivel de sim_default do locutor (media dos 3 treinos),
# que e endogeno (medido na saida), e o f0 da referencia (exogeno).
cs = W.groupby("spk").sim_default.mean().rename("c_hat")
f0 = W.groupby("spk").ref_f0_hz.first()
S = W.groupby(["spk", "N"]).agg(q=("copied", lambda v: 1 - v.mean()), m=("m", "mean"), sim_ref=("sim_ref", "mean"),
                                 sim_def=("sim_default", "mean"), sim_spk=("sim_spk", "mean"), ms=("ms", "mean")).reset_index()
S = S.merge(cs, left_on="spk", right_index=True).merge(f0.rename("f0"), left_on="spk", right_index=True)
print("extrapolacao nas 38 externas: media %.3f dp %.3f (nao usada)" % (cs_ex.mean(), cs_ex.std()))
print(f"\nc_hat (38 externas): media {cs.mean():.3f}, dp {cs.std():.3f}, intervalo [{cs.min():.3f}, {cs.max():.3f}]; corr(c_hat, f0) Spearman = {spearmanr(cs, f0.loc[cs.index])[0]:.2f}")
print("sim_default medio por locutor (media dos 3 treinos) vs c_hat: Spearman", round(spearmanr(S.groupby('spk').sim_def.mean(), cs.sort_index())[0], 2))

# ---------- 2. quanto da falha por locutor e explicado por proximidade a voz padrao vs N ----------
print("\n=== decomposicao da variancia (38 locutores x 3 N) ===")
for col in ["q", "m", "sim_ref", "sim_spk"]:
    Y = S.pivot(index="spk", columns="N", values=col)
    gm = Y.values.mean(); ssN = 38 * ((Y.mean(axis=0) - gm) ** 2).sum(); ssS = 3 * ((Y.mean(axis=1) - gm) ** 2).sum(); ssT = ((Y.values - gm) ** 2).sum()
    sp = Y.mean(axis=1)
    covs = pd.DataFrame({"f0": f0.loc[sp.index], "c_hat": cs.loc[sp.index]})
    def r2(cols):
        X = np.c_[np.ones(len(sp)), covs[cols].values]; b = np.linalg.lstsq(X, sp.values, rcond=None)[0]; e = sp.values - X @ b
        return 1 - (e ** 2).sum() / ((sp.values - sp.mean()) ** 2).sum()
    print(f"{col:8s}: fracao do SS total: N {ssN/ssT:.2f}, locutor {ssS/ssT:.2f}, resto {(ssT-ssN-ssS)/ssT:.2f} | do efeito de locutor, R2 explicado por f0 {r2(['f0']):.2f}, c_hat {r2(['c_hat']):.2f}, ambos {r2(['f0','c_hat']):.2f}")
print("\nSpearman por N (taxa de copia 1-q, margem) com f0 e c_hat:")
for n_, x in S.groupby("N"):
    print(f"  N={n_}: copia~f0 {spearmanr(1-x.q, x.f0)[0]:+.2f}, copia~c_hat {spearmanr(1-x.q, x.c_hat)[0]:+.2f}, margem~c_hat {spearmanr(x.m, x.c_hat)[0]:+.2f}, "
          f"sim_ref~c_hat {spearmanr(x.sim_ref, x.c_hat)[0]:+.2f}, sim_spk~c_hat {spearmanr(x.sim_spk, x.c_hat)[0]:+.2f}, sim_ref~f0 {spearmanr(x.sim_ref, x.f0)[0]:+.2f}, sim_def~f0 {spearmanr(x.sim_def, x.f0)[0]:+.2f}")
# modelo amostra a amostra: logit da copia ~ ln N + c_hat (+ f0): quanto do desvio explica cada termo
from sklearn.linear_model import LogisticRegression
W2 = W.merge(cs, left_on="spk", right_index=True); W2["lnN"] = np.log(W2.N); W2["f0z"] = (W2.ref_f0_hz - W2.ref_f0_hz.mean()) / W2.ref_f0_hz.std()
W2["cz"] = (W2.c_hat - W2.c_hat.mean()) / W2.c_hat.std()
print("\nregressao logistica amostra a amostra (sem penalizacao); pseudo-R2 de McFadden (c_hat = sim_default medio do locutor):")
y = W2.copied.values; ll0 = len(y) * (y.mean() * np.log(y.mean()) + (1 - y.mean()) * np.log(1 - y.mean()))
def mcf(X):
    r = LogisticRegression(C=1e6, max_iter=5000).fit(X, y); p = np.clip(r.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return 1 - (y * np.log(p) + (1 - y) * np.log(1 - p)).sum() / ll0, r.coef_[0]
for f in [["lnN"], ["cz"], ["f0z"], ["lnN", "cz"], ["lnN", "f0z"], ["lnN", "cz", "f0z"]]:
    pr2, co = mcf(W2[f].values); print(f"  {'+'.join(f):12s} pseudoR2={pr2:.3f} coefs={np.round(co, 2)}")
Xd = pd.get_dummies(W2.spk, dtype=float).values; pr2, _ = mcf(np.c_[Xd, W2.lnN.values])
print(f"  lnN + locutor (fator) pseudoR2={pr2:.3f} (teto do que propriedades fixas da voz explicam)")

# ---------- 3. curva em N com metricas alternativas; expoente pela razao de ganhos ----------
print("\n=== curva em N, metricas alternativas (media sobre os 38 locutores) e expoente pela razao de ganhos ===")
print("alpha_g = log4[(y32 - y8)/(y128 - y32)] vale para y = y_inf - A N^-alpha com piso livre; alpha_0 = -log4(q128/q8)/2 para q = c N^-alpha sem piso")
mets = {"copia (1-q)": lambda x: 1 - x.q, "margem m": lambda x: x.m, "sim_ref": lambda x: x.sim_ref, "sim_spk": lambda x: x.sim_spk,
        "-sim_default": lambda x: -x.sim_def, "margem por locutor ms": lambda x: x.ms, "copia por sim_spk > sim_default": lambda x: x.cop_spk}
S2 = S.merge(W2.groupby(["spk", "N"]).copied_spk.mean().rename("cop_spk").reset_index(), on=["spk", "N"])
spk = np.array(sorted(S.spk.unique()))
def curves(sub):
    res = {}
    for nm, f in mets.items():
        Y = sub.assign(y=f(sub)).pivot(index="spk", columns="N", values="y").loc[:, [8, 32, 128]].values
        res[nm] = Y
    return res
def alpha_g(y): 
    g1, g2 = y[1] - y[0], y[2] - y[1]
    return np.log(g1 / g2) / np.log(4) if (g1 > 0 and g2 > 0) else np.nan
rows = []
for subset, sel in [("todos (38)", spk), ("saida longe da voz padrao (sim_default medio < mediana, 19)", spk[cs.loc[spk].values < np.median(cs)]),
                    ("saida perto da voz padrao (sim_default medio >= mediana, 19)", spk[cs.loc[spk].values >= np.median(cs)]),
                    ("f0 < mediana (graves, 19)", spk[f0.loc[spk].values < np.median(f0)]), ("f0 >= mediana (agudas, 19)", spk[f0.loc[spk].values >= np.median(f0)])]:
    C = curves(S2[S2.spk.isin(sel)])
    for nm, Y in C.items():
        y = Y.mean(axis=0); bs = np.array([alpha_g(Y[rng.integers(0, len(Y), len(Y))].mean(0)) for _ in range(B)])
        ok = np.isfinite(bs)
        row = dict(subconjunto=subset, metrica=nm, y8=y[0], y32=y[1], y128=y[2], ganho_8_32=y[1] - y[0], ganho_32_128=y[2] - y[1],
                   alpha_g=alpha_g(y), lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5), P_alpha_maior_0_5=np.mean(bs[ok] > 0.5), frac_indef=1 - ok.mean())
        if nm.startswith("copia"):
            qq = 1 - y; row["alpha0"] = -np.log(qq[2] / qq[0]) / np.log(16)
        rows.append(row)
A = pd.DataFrame(rows); A.to_csv(os.path.join(out, "artefato_expoentes.csv"), index=False)
print(A.round(3).to_string())
S2.to_csv(os.path.join(out, "artefato_por_locutor.csv"), index=False)

# ---------- 4. o mesmo expoente acrescentando ruido de semente (DP do efeito principal de semente medido nas 4 vozes novas) ----------
import json
sd = json.load(open(os.path.join(out, "seed_sd.json")))
print("\nalpha_g com reamostragem de locutores + ruido de semente N(0, sd^2) em cada um dos 3 treinos:")
for nm, sdv in [("copia (1-q)", sd["unseen|copied"]["sd_seed_main"]), ("margem m", sd["unseen|m"]["sd_seed_main"])]:
    Y = curves(S2)[nm]
    for s_ in [0.0, sdv / 2, sdv]:
        bs = np.array([alpha_g(Y[rng.integers(0, 38, 38)].mean(0) + rng.normal(0, s_, 3)) for _ in range(B)])
        ok = np.isfinite(bs)
        print(f"  {nm:12s} sd_semente={s_:.4f}: IC95 [{np.nanpercentile(bs,2.5):.2f}, {np.nanpercentile(bs,97.5):.2f}], ganho 32->128 <= 0 ou indefinido em {1-ok.mean():.2f}, P(alpha>0.5)={np.mean(bs[ok]>0.5):.2f}, P(alpha<0)={np.mean(bs[ok]<0):.2f}")
