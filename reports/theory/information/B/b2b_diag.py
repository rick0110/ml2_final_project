"""b2b: diagnostico (NAO e previsao): varre s2 e M_eff e mostra a curva externa prevista; e ajuste com M_eff fixo = 13100 (LJSpeech inteiro).
Uso: python3 -I b2b_diag.py out/b0.npz"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize_scalar
from bt_lib import *
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "b2_core.py")).read())
WIDE = ["n8_k32_s1-wide", "n32_k32_s1-wide", "n128_k32_s1-wide"]
print("s2 \\ M_eff : q_ext(8,32,128) link=samp  [obs .43 .30 .21]")
for Me in [2000, 13100, 50000]:
    for s2 in [5, 10, 20, 40, 80]:
        q = []
        for r in WIDE:
            g = GEO[r]; k, n, idx = obs(g, "external"); q.append(1 - p_copy(lam_run(g, s2, np.log(Me))[idx], "samp").mean())
        print(f"M={Me:6d} s2={s2:4d}: " + " ".join(f"{x:.2f}" for x in q))
o = minimize_scalar(lambda ls: nll([ls, np.log(13100)], "seen", "samp", True, BASE), bounds=(0, 6), method="bounded")
print("M_eff=13100 fixed: s2=%.2f nll_seen=%.1f" % (np.exp(o.x), o.fun))
for r in BASE:
    g = GEO[r]; L = lam_run(g, np.exp(o.x), np.log(13100)); s = f"{r:18s}"
    for grp in ["seen", "unseen", "external"]:
        k, n, idx = obs(g, grp)
        if len(k): s += f" | {grp[:3]} obs {1-k.sum()/n.sum():.2f} pred {1-p_copy(L[idx],'samp').mean():.2f}"
    print(s)
