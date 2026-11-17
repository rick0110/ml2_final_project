"""b1: exploracao: Lambda com T=1 (sem parametro) por treino e grupo. Uso: python3 -I b1_explora.py out/b0.npz"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bt_lib import *
S = load(sys.argv[1])
BASE = [r for r in S if S[r]["var"] in ("", "wide") and S[r]["refs"]]
for T in [1, 3, 10, 30]:
    print("T =", T)
    for run in sorted(BASE, key=lambda r: (S[r]["N"], S[r]["K"], r)):
        g = Geo(S[run]); L, parts = lam(g, T=T, return_parts=True)
        line = f"{run:20s}"
        for grp in ["seen", "unseen", "external"]:
            k, n, idx = obs(g, grp)
            if len(k) == 0: continue
            qo = 1 - k.sum() / n.sum(); qm = 1 - p_copy(L[idx], "mean").mean(); qs = 1 - p_copy(L[idx], "samp").mean()
            line += f" | {grp[:3]} qobs={qo:.2f} qmean={qm:.2f} qsamp={qs:.2f} Lmed={np.median(L[idx]):6.1f} Lsd={L[idx].std():5.1f}"
        print(line)
