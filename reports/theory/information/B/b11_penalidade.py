"""b11: penalidade de generalizacao exigida: deslocamento uniforme Delta_gen (nats-contagem) que leva a lei calibrada nas treinadas (b8)
a reproduzir q das externas em cada N; teste da forma Delta_gen ~ N^-1/2 (cota de informacao mutua). Uso: python3 -I b11_penalidade.py out/b0.npz"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import norm, linregress
from scipy.optimize import brentq
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "b8_lei.py")).read().split("res = dict(")[0].replace("print(f\"ln M_eff", "_ = (f\"ln M_eff"))
out = []
for r in ["n8_k32_s1-wide", "n32_k32_s1-wide", "n128_k32_s1-wide"]:
    for mode in ["nw", "nn"]:
        qo, _, L, _ = q_of(r, "external", lnM, mode)
        d = brentq(lambda s: (1 - norm.cdf((L - s) / TAU).mean()) - qo, -20, 40)
        out.append((G[r]["g"].N, mode, d)); print(f"  {r} cobertura={mode}: Delta_gen = {d:.2f} nats-contagem ({d/TAU:.2f} tau)")
for mode in ["nw", "nn"]:
    a = np.array([(n, d) for n, m, d in out if m == mode])
    lr = linregress(np.log(a[:, 0]), np.log(a[:, 1])); print(f"  {mode}: Delta_gen ~ N^-b, b = {-lr.slope:.2f} (3 pontos)")
