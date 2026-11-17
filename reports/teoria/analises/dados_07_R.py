"""Tarefa 7: efeito de R (media dos codigos de 4/16 frases de referencia) com estatistica pareada por amostra.
Bootstrap por cluster: reamostra locutores do grupo (os mesmos em todos os treinos) e frases. Uso: python3 -I dados_07_R.py <results_dir> <out_dir> [B]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from dados_comum import load
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
df = load(sys.argv[1]); out = sys.argv[2]; B = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
rng = np.random.default_rng(77)
d = df[df.strength == 1.0]
runsR = sorted(d[d.R > 1].run.unique())
cols = ["copied", "copied_spk", "m", "ms", "sim_ref", "sim_spk", "sim_default", "wer"]
key = ["run", "group", "spk", "sent"]
base = d[(d.R == 1) & d.run.isin(runsR)].set_index(key)[cols]
rows = []; per_run = []
for R in [4, 16]:
    alt = d[d.R == R].set_index(key)[cols]
    J = base.join(alt, lsuffix="_1", rsuffix="_R", how="inner")
    for c in cols: J["d_" + c] = J[c + "_R"] - J[c + "_1"]
    J = J.reset_index()
    for (grp, setname), x in J.assign(setname=np.where(J.run.str.endswith("-wide"), "wide", "plain")).groupby(["group", "setname"]):
        runs = sorted(x.run.unique()); spks = np.array(sorted(x.spk.unique()))
        # matriz treino x locutor de medias das diferencas (frases como replicas)
        M = {c: x.pivot_table(index="spk", columns="run", values="d_" + c, aggfunc="mean").loc[spks, runs].values for c in cols}
        for c in cols:
            est = np.nanmean(M[c])
            bs = np.array([np.nanmean(M[c][rng.integers(0, len(spks), len(spks))]) for _ in range(B)])
            spk_mean = np.nanmean(M[c], 1)
            try: pw = wilcoxon(spk_mean[spk_mean != 0]).pvalue if (spk_mean != 0).sum() >= 5 else np.nan
            except ValueError: pw = np.nan
            rows.append(dict(R=R, grupo=grp, conjunto=setname, metrica=c, n_treinos=len(runs), n_loc=len(spks), dif=est, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                             loc_sobe=int((spk_mean > 0).sum()), loc_desce=int((spk_mean < 0).sum()), p_wilcoxon_loc=pw,
                             treinos_sobe=int((np.nanmean(M[c], 0) > 0).sum()), base=x[c + "_1"].mean()))
        for r_, xr in x.groupby("run"):
            per_run.append(dict(R=R, run=r_, grupo=grp, **{c: xr[c + "_1"].mean() for c in ["copied", "copied_spk", "wer"]},
                                **{c + "_R": xr[c + "_R"].mean() for c in ["copied", "copied_spk", "wer"]}))
T = pd.DataFrame(rows); T.to_csv(os.path.join(out, "efeito_R.csv"), index=False)
print("diferenca media (R - R1), IC95 bootstrap de locutores (pareado), n locutores que sobem/descem, Wilcoxon sobre medias por locutor")
print(T[T.metrica.isin(["copied", "copied_spk", "m", "ms", "sim_spk", "sim_default", "wer"])].round(4).to_string())
print(pd.DataFrame(per_run).round(3).to_string())
