"""b3: auxiliary measured facts (no model): WER x copy, duration x margin noise, R=4 per voice, strength 0.6.
Usage: python3 -I b3_fatos.py out/b0.npz"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import spearmanr, linregress
from bt_lib import *
S = load(sys.argv[1])
BASE = sorted([r for r in S if S[r]["var"] in ("", "wide") and S[r]["refs"]], key=lambda r: (S[r]["N"], S[r]["K"], r))
print("== 1. WER x copy per voice (within run), and mean WER per group")
for r in BASE:
    s = f"{r:18s}"
    for grp in ["seen", "unseen", "external"]:
        R = [x for x in S[r]["refs"] if x["grp"] == grp]
        if not R: continue
        c = np.array([x["r1"]["k"] / x["r1"]["n"] for x in R]); w = np.array([x["r1"]["wer"] for x in R])
        s += f" | {grp[:3]} WER {w.mean():.3f}" + (f" rho(copy,WER) {spearmanr(c, w).correlation:+.2f}" if len(R) > 6 else "")
    print(s)
# per-sample within-voice: WER of copied vs not copied samples (voices with 0<p<1)
d_c, d_n = [], []
for r in BASE:
    for x in S[r]["refs"]:
        pc = np.array(x["r1"]["per_copy"]); pw = np.array(x["r1"]["per_wer"])
        if 0 < pc.mean() < 1: d_c.append(np.nanmean(pw[pc == 1])); d_n.append(np.nanmean(pw[pc == 0]))
d_c, d_n = np.array(d_c), np.array(d_n)
print(f"samples within voice (mixed voices, n={len(d_c)}): WER copied {np.nanmean(d_c):.3f} vs not copied {np.nanmean(d_n):.3f}; mean diff {np.nanmean(d_c-d_n):+.3f} (se {np.nanstd(d_c-d_n)/np.sqrt(len(d_c)):.3f})")
print("\n== 2. Large deviations: within-voice SD of the margin vs duration of the synthesised utterance")
sl = []; allm = []; alld = []
for r in BASE:
    for x in S[r]["refs"]:
        m = np.array(x["r1"]["per_m"]); du = np.array(x["r1"]["per_dur"])
        res = m - m.mean(); allm += list(res); alld += list(du)
allm, alld = np.array(allm), np.array(alld)
qs = np.quantile(alld, [0, .2, .4, .6, .8, 1])
for a, b in zip(qs[:-1], qs[1:]):
    sel = (alld >= a) & (alld <= b)
    print(f"  duration {a:4.1f}-{b:4.1f}s: residual SD {allm[sel].std():.4f}  n={sel.sum()}")
lr = linregress(np.log(alld[alld > 0]), np.log(np.abs(allm[alld > 0]) + 1e-6))
bins = [(alld >= a) & (alld <= b) for a, b in zip(qs[:-1], qs[1:])]
sds = np.array([allm[s].std() for s in bins]); mids = np.array([np.median(alld[s]) for s in bins])
lr2 = linregress(np.log(mids), np.log(sds))
print(f"  slope log SD vs log duration (5 quintiles): {lr2.slope:.3f} +- {lr2.stderr:.3f}  (Chernoff/CLT predicts -0.5 if the margin is a mean of iid frames)")
print("\n== 3. R=4 per voice (externals, wide; sim_spk criterion in both)")
for r in ["n8_k32_s1-wide", "n32_k32_s1-wide", "n128_k32_s1-wide"]:
    R = [x for x in S[r]["refs"] if x["grp"] == "external" and "r4" in x]
    p1 = np.array([x["r1"]["k_spk"] / x["r1"]["n"] for x in R]); p4 = np.array([x["r4"]["k_spk"] / x["r4"]["n"] for x in R])
    q1 = np.array([x["r1"]["k"] / x["r1"]["n"] for x in R])
    flip = np.mean((p1 > .5) != (p4 > .5)); w1 = np.mean([x["r1"]["wer"] for x in R]); w4 = np.mean([x["r4"]["wer"] for x in R])
    mid = (p1 > .1) & (p1 < .9)
    print(f"  {r}: copy(spk) R1 {p1.mean():.3f} R4 {p4.mean():.3f}; mean |diff| per voice {np.abs(p4-p1).mean():.3f}; majority flips {flip:.3f}; "
          f"|diff| on extreme voices {np.abs(p4-p1)[~mid].mean():.3f} vs intermediate {np.abs(p4-p1)[mid].mean():.3f} (n_int={mid.sum()}); WER {w1:.3f}->{w4:.3f}; corr(p1,p4)={np.corrcoef(p1,p4)[0,1]:.2f}")
print("\n== 4. Strength 0.6 vs 1.0 (plain runs with strength 0.6)")
agg = {}
for r in BASE:
    for x in S[r]["refs"]:
        if "s06" not in x: continue
        a = agg.setdefault(x["grp"], dict(c1=[], c6=[], w1=[], w6=[], m1=[], m6=[], sd1=[], sd6=[]))
        a["c1"].append(x["r1"]["k"] / x["r1"]["n"]); a["c6"].append(x["s06"]["k"] / x["s06"]["n"])
        a["w1"].append(x["r1"]["wer"]); a["w6"].append(x["s06"]["wer"]); a["m1"].append(x["r1"]["m"]); a["m6"].append(x["s06"]["m"])
        a["sd1"].append(x["r1"]["sim_def"]); a["sd6"].append(x["s06"]["sim_def"])
for g, a in agg.items():
    a = {k: np.array(v) for k, v in a.items()}
    print(f"  {g:8s} n={len(a['c1'])}: copy {a['c1'].mean():.3f}->{a['c6'].mean():.3f}; WER {a['w1'].mean():.3f}->{a['w6'].mean():.3f} "
          f"(diff {np.mean(a['w6']-a['w1']):+.3f} se {np.std(a['w6']-a['w1'])/np.sqrt(len(a['w1'])):.3f}); margin {a['m1'].mean():+.3f}->{a['m6'].mean():+.3f}; sim_default {a['sd1'].mean():.3f}->{a['sd6'].mean():.3f}; "
          f"margin ratio m6/m1 (voices with m1>0.03) {np.median(a['m6'][a['m1']>0.03]/a['m1'][a['m1']>0.03]):.2f}")
