import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dados_comum import load
df = load(sys.argv[1])
d = df[(df.strength==1.0)&(df.R==1)]
for g in ["seen","unseen","external"]:
    s = d[d.group==g].groupby("run").reference.apply(lambda x: tuple(sorted(set(x))))
    print(g, s.value_counts().to_dict() if g!="external" else s.map(len).to_dict())
z = df[df.strength==0.0]
print(z.groupby(["run","group"]).agg(n=("m","size"), spk=("spk","nunique"), sim_ref=("sim_ref","mean"), sdef=("sim_default","mean"), cop=("copied","mean")))
print(sorted(z[z.group=="unseen"].spk.unique()), sorted(d[(d.run=="n8_k32_s1-wide")&(d.group=="unseen")].spk.unique()))
w = d[(d.run=="n8_k32_s1-wide")&(d.group=="external")]
print(w.spk.nunique(), w.groupby("spk").sent.nunique().unique(), w.ref_f0_hz.isna().sum())
print(df.groupby(["run","strength","R"]).size().to_string())
