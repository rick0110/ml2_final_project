"""b27: (a) real speech among the 8 trained voices (chain); (b) outputs of the 8 trained voices in the wide runs (Fano); (c) C at strength 0.0 (ECAPA only) in the -spk/-spkdrop runs.
Usage: python3 -I b27_extra.py out/b0.npz out/b12_emb.npz out/b20_outs.npz out/b20_refs.npz"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bt_lib import load
S = load(sys.argv[1])
def ld(p):
    d = np.load(p, allow_pickle=False); E = d["emb"] / np.linalg.norm(d["emb"], axis=1, keepdims=True); return list(d["keys"].astype(str)), E
OUT, oth = {}, {}
for f in sys.argv[2:5]:
    ks, E = ld(f)
    for i, k in enumerate(ks):
        p = k.split("|")
        if p[0] == "out": ref, s = p[3].rsplit("__", 1); OUT.setdefault((p[1], p[2]), {}).setdefault(ref, []).append(E[i])
        elif p[0] == "oth": oth.setdefault(p[1], []).append(E[i])
def fano(acc, n):
    pe = min(max(1 - acc, 1e-12), 1 - 1e-12); return np.log2(n) - (-(pe*np.log2(pe)+(1-pe)*np.log2(1-pe))) - pe*np.log2(n-1)
refs = S["n8_k128_s1-spk"]["refs"]; SEEN = sorted(x["ref"] for x in refs if x["grp"] == "seen"); V12 = sorted(x["ref"] for x in refs if x["grp"] in ("seen", "unseen"))
def real_id(vs):
    spk = [v.split("_")[0] for v in vs]; hits = []
    for j, s in enumerate(spk):
        for i in range(len(oth[s])):
            C = []
            for s2 in spk:
                L = [e for t, e in enumerate(oth[s2]) if not (s2 == s and t == i)]; c = np.mean(L, 0); C.append(c / np.linalg.norm(c))
            hits.append(np.argmax(np.array(C) @ oth[s][i]) == j)
    return np.mean(hits), len(hits)
a, n = real_id(SEEN); print(f"(a) real speech, 8 trained: accuracy {a:.3f} (n={n}) Fano {fano(a, 8):.2f} / 3")
a, n = real_id(V12); print(f"    real speech, 12 (V12):  accuracy {a:.3f} (n={n}) Fano {fano(a, 12):.2f} / 3.58")
def out_id(run, cond, vs):
    o = OUT.get((run, cond), {})
    if not all(v in o for v in vs): return None
    C = np.array([np.mean(oth[v.split("_")[0]], 0) for v in vs]); C /= np.linalg.norm(C, axis=1, keepdims=True)
    hits = [np.argmax(C @ e) == j for j, v in enumerate(vs) for e in o[v]]; return np.mean(hits), len(hits)
for run in ["n8_k32_s1-wide", "n128_k32_s1-wide", "n128_k32_s1", "n200_k32_s1", "n8_k128_s1-spk", "n32_k128_s1-spkdrop"]:
    r = out_id(run, "strength_1.0", SEEN)
    if r: print(f"(b) output, 8 trained, {run}: accuracy {r[0]:.3f} Fano {fano(r[0], 8):.2f}")
print("(c) strength 0.0 (no VAE style; in the -spk runs only the ECAPA vector remains) x 1.0, V12, accuracy/Fano")
for run in ["n2_k128_s1-spk", "n8_k8_s1-spk", "n8_k128_s1-spk", "n8_k128_s1-spkdrop", "n32_k8_s1-spkdrop", "n32_k128_s1-spkdrop"]:
    vs = [v for v in V12 if v in OUT.get((run, "strength_0.0"), {})]
    r0 = out_id(run, "strength_0.0", vs); r1 = out_id(run, "strength_1.0", vs)
    if r0 and r1: print(f"    {run:20s} ({len(vs)} voices): strength 0.0 accuracy {r0[0]:.3f} Fano {fano(r0[0], len(vs)):.2f} | strength 1.0 accuracy {r1[0]:.3f} Fano {fano(r1[0], len(vs)):.2f}")
