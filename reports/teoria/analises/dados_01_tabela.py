import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from dados_comum import load, main_cond
df = load(sys.argv[1]); out = sys.argv[2]
d = main_cond(df)
g = d.groupby(["variant", "N", "K", "seed", "run", "group"])
t = g.agg(n=("m", "size"), n_spk=("spk", "nunique"), copied=("copied", "mean"), wer=("wer", "mean"),
          m_mean=("m", "mean"), m_sd=("m", "std"), ms_mean=("ms", "mean"), ms_sd=("ms", "std"),
          copied_spk=("copied_spk", "mean"), sim_ref=("sim_ref", "mean"), sim_default=("sim_default", "mean")).reset_index()
t.to_csv(os.path.join(out, "tabela_runs.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(t.round(3).to_string())
print(d.groupby('run').sent.nunique().unique(), d.wer.describe())
