"""Shared loader. Usage: imported by the dados_*.py scripts (which take the results directory as an argument)."""
import re, glob, os
import numpy as np, pandas as pd

def load(results_dir, step=2500):
    rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, f"*/step_{step}/strength_*/results.csv"))):
        parts = p.split(os.sep)
        run, cond = parts[-4], parts[-2]
        m = re.fullmatch(r"n(\d+)_k(\d+)_s(\d+)(?:-(\w+))?", run)
        c = re.fullmatch(r"strength_([\d.]+)(?:_r(\d+))?", cond)
        d = pd.read_csv(p)
        d["run"] = run; d["N"] = int(m.group(1)); d["K"] = int(m.group(2)); d["seed"] = int(m.group(3))
        d["variant"] = m.group(4) or "plain"
        d["strength"] = float(c.group(1)); d["R"] = int(c.group(2) or 1)
        if "sim_spk" not in d: d["sim_spk"] = np.nan
        d["group"] = d["reference_emotion"]
        d["m"] = d.sim_ref - d.sim_default
        d["ms"] = d.sim_spk - d.sim_default
        d["copied"] = (d.sim_ref > d.sim_default).astype(float)
        d["copied_spk"] = np.where(d.sim_spk.isna(), np.nan, (d.sim_spk > d.sim_default).astype(float))
        d["sent"] = d["sample"].str.extract(r"__s(\d+)$")[0].astype(int)
        d["spk"] = d["reference_speaker"].astype(str)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)

def main_cond(df):
    return df[(df.strength == 1.0) & (df.R == 1)]
