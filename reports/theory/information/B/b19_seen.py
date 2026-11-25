"""b19: output-output capacity (R=1 x R=4 of the same reference) per group, N=8 vs N=128 (wide). Usage: python3 -I b19_seen.py out/b0.npz out/b12_emb.npz"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bt_lib import load
S = load(sys.argv[1]); d = np.load(sys.argv[2], allow_pickle=False); rng = np.random.default_rng(8)
keys = d["keys"].astype(str); E = d["emb"] / np.linalg.norm(d["emb"], axis=1, keepdims=True); K_ = [k.split("|") for k in keys]
def get(run, cond): return {k[3].rsplit("__", 1)[0]: E[i] for i, k in enumerate(K_) if k[0] == "out" and k[1] == run and k[2] == cond}
for grp, n in [("seen", 8), ("external", 38)]:
    for N, run in [(8, "n8_k32_s1-wide"), (128, "n128_k32_s1-wide")]:
        ids = [x["ref"] for x in S[run]["refs"] if x["grp"] == grp]
        if grp == "external": ids = list(rng.permutation(ids))
        a, b = get(run, "strength_1.0"), get(run, "strength_1.0_r4")
        accs = []
        for rep in range(200 if grp == "external" else 1):
            sub = ids if grp == "seen" else list(rng.permutation(ids)[:8])   # same 8 candidates for comparison
            A = np.array([a[i] for i in sub]); B = np.array([b[i] for i in sub]); accs.append(np.mean((A @ B.T).argmax(1) == np.arange(len(sub))))
        print(f"  {grp:8s} N={N:3d}: accuracy R1->R4 among 8 voices = {np.mean(accs):.3f}" + (" (mean over 200 subsets of 8)" if grp == "external" else ""))
