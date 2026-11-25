"""b18: a single mechanism (capacity): capacity change measured in the audio -> shift of the per-voice state.
 beta (state sd per bit) = gamma / (dC/dlnN)  [gamma from the minimal model b16; C(N) from b13]; t from b16.
 Predicted: copying at strength 0.6 (dC measured output-output in the plain runs) and mean effect of R=4 (fidelity dC, wide).
Usage: python3 -I b18_capshift.py out/b0.npz out/b13.json out/b16.json"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import norm
from bt_lib import load
S = load(sys.argv[1]); R = json.load(open(sys.argv[2])); M = json.load(open(sys.argv[3])); t, gam = M["t"], M["gamma"]
C = {"ce": (0.75, 2.01), "fano": (0.31, 1.13)}
dC6 = {e: np.mean([R[f"plain|{r}|strength_0.6"][e] - R[f"plain|{r}|strength_1.0"][e] for r in ["n128_k8_s1", "n128_k32_s1", "n200_k32_s1"]]) for e in C}
dCR4 = {"ce": {8: -0.18, 128: -0.41}, "fano": {8: R["n8_k32_s1-wide|strength_1.0_r4|cent"]["I_fano"] - R["n8_k32_s1-wide|strength_1.0|cent"]["I_fano"],
                                            128: R["n128_k32_s1-wide|strength_1.0_r4|cent"]["I_fano"] - R["n128_k32_s1-wide|strength_1.0|cent"]["I_fano"]}}
for e, (c8, c128) in C.items():
    beta = gam / ((c128 - c8) / np.log(16)); dmu = beta * dC6[e]
    print(f"[{e}] beta = {beta:.2f} sd/bit; strength 0.6: dC = {dC6[e]:+.2f} bits -> dmu = {dmu:+.2f} sd")
    agg = {}
    for r in sorted(S):
        if S[r]["var"] != "" or not S[r]["refs"]: continue
        for x in S[r]["refs"]:
            if "s06" not in x: continue
            L = t * norm.ppf((x["r1"]["k"] + .5) / 26); a = agg.setdefault(x["grp"], [[], [], []])
            a[0].append(x["r1"]["k"] / 25); a[1].append(x["s06"]["k"] / 25); a[2].append(norm.cdf((L + dmu) / t))
    for gname, a in agg.items():
        a = np.array(a); print(f"    {gname:8s}: copy 1.0 {a[0].mean():.3f} | 0.6 obs {a[1].mean():.3f} pred {a[2].mean():.3f}")
    for N, run in [(8, "n8_k32_s1-wide"), (128, "n128_k32_s1-wide")]:
        X = [x for x in S[run]["refs"] if x["grp"] == "external" and "r4" in x]
        p1 = np.array([x["r1"]["k_spk"] / 25 for x in X]); p4 = np.array([x["r4"]["k_spk"] / 25 for x in X])
        L = t * norm.ppf((p1 * 25 + .5) / 26); pr = norm.cdf((L + beta * dCR4[e][N]) / t)
        print(f"    R=4, N={N}: dC = {dCR4[e][N]:+.2f} bits -> dcopy pred {pr.mean()-p1.mean():+.3f} obs {p4.mean()-p1.mean():+.3f}")
