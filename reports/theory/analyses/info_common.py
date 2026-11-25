"""Shared helpers (no imports from data dirs). Run scripts with python3 -I and add this dir to sys.path explicitly."""
import re, os, csv, json, numpy as np
LOG2 = np.log(2.0)
def parse_run(name):
    m = re.match(r"n(\d+)_k(\d+)_s(\d+)(?:-(.*))?$", name)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4) or "") if m else None
def list_runs(root, step="step_2500"):
    out = []
    for r in sorted(os.listdir(root)):
        p = parse_run(r)
        if p and os.path.exists(os.path.join(root, r, step, "latents.npz")):
            out.append((r,) + p)
    out.sort(key=lambda t: (t[4] != "", t[4], t[1], t[2], t[3]))
    return out
def load_latents(root, run, step="step_2500"):
    z = np.load(os.path.join(root, run, step, "latents.npz"), allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["mu"] = d["mu"].astype(np.float64); d["logvar"] = d["logvar"].astype(np.float64)
    d["train"] = d["source"] != "refs"
    return d
def load_results(root, run, sub="strength_1.0", step="step_2500"):
    p = os.path.join(root, run, step, sub, "results.csv")
    if not os.path.exists(p): return []
    with open(p, encoding="utf-8") as f: return list(csv.DictReader(f))
def kl_bits(mu, logvar):
    """per-dim KL(N(mu,exp(logvar))||N(0,1)) in bits, shape (n,d)"""
    return 0.5 * (mu**2 + np.exp(logvar) - logvar - 1.0) / LOG2
def entropy_bits(counts):
    p = np.asarray(counts, float); p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())
def pr(lam):
    lam = np.clip(np.asarray(lam, float), 0, None)
    return float(lam.sum()**2 / (lam**2).sum()) if lam.sum() > 0 else float("nan")
def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n"
    for r in rows: s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s
