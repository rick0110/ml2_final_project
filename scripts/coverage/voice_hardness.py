"""Per-voice difficulty in the -wide runs: is it stable across N, and what predicts it?

Usage: python scripts/coverage/voice_hardness.py <folder with the run folders (results.csv and latents.npz)>
"""
import sys, csv, numpy as np, collections
from scipy.stats import spearmanr
base=sys.argv[1]
runs={8:"n8_k32_s1-wide",32:"n32_k32_s1-wide",128:"n128_k32_s1-wide"}
rate={}; f0={}
for n,r in runs.items():
    d=collections.defaultdict(list)
    for row in csv.DictReader(open(f"{base}/{r}/step_2500/strength_1.0/results.csv")):
        if row["reference_emotion"]=="external":
            d[row["reference_speaker"]].append(float(row["sim_ref"])>float(row["sim_default"])); f0[row["reference_speaker"]]=float(row["ref_f0_hz"] or "nan")
    rate[n]={s:np.mean(v) for s,v in d.items()}
spk=sorted(set.intersection(*[set(v) for v in rate.values()]))
print("speakers",len(spk))
for a,b in [(8,32),(32,128),(8,128)]:
    print(f"rank corr of per-speaker copy rate N={a} vs N={b}: {spearmanr([rate[a][s] for s in spk],[rate[b][s] for s in spk])[0]:.2f}")
print("corr copy rate vs ref F0 (default voice is female):",{n:round(spearmanr([rate[n][s] for s in spk],[f0[s] for s in spk])[0],2) for n in runs})
for n,r in runs.items():
    z=np.load(f"{base}/{r}/step_2500/latents.npz",allow_pickle=True)
    mu,sp,em=z["mu"],z["speaker"],z["emotion"]
    tr=(em!="seen")&(em!="unseen")&(em!="external")
    means={s:mu[tr&(sp==s)].mean(0) for s in set(sp[tr])}
    lj=[s for s in means if not s.isdigit()]
    libri=np.array([means[s] for s in means if s.isdigit()]); dflt=np.mean([means[s] for s in lj],0) if lj else None
    dn=[];dd=[];cr=[]
    for s in spk:
        x=mu[(em=="external")&(sp==s)]
        if len(x)==0: continue
        x=x.mean(0); dn.append(np.min(np.linalg.norm(libri-x,axis=1))); dd.append(np.linalg.norm(x-dflt) if dflt is not None else np.nan); cr.append(rate[n][s])
    print(f"N={n}: train speakers {len(libri)}, default ids {lj[:3]}; corr(copy, dist to nearest trained voice)={spearmanr(cr,dn)[0]:.2f}; corr(copy, dist to default voice)={spearmanr(cr,dd)[0]:.2f}; median nearest dist {np.median(dn):.2f}")
