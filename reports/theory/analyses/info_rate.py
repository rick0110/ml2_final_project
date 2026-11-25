"""Task 1: rate KL(q||p) per dim / total in bits for every run. usage: python3 -I info_rate.py DATA OUT"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from info_common import *
root, out = sys.argv[1], sys.argv[2]
rows = []; res = {}
rng = np.random.default_rng(0)
for run, N, K, seed, var in list_runs(root):
    d = load_latents(root, run); tr = d["train"]; lj = tr & (d["speaker"] == "lj"); lib = tr & ~lj
    kl = kl_bits(d["mu"], d["logvar"])            # (n,32)
    per_dim = kl[tr].mean(0)
    mu, s2 = d["mu"][tr], np.exp(d["logvar"][tr])
    # decomposition of each dim's rate: -0.5 log s2 -0.5 (precision term), 0.5 E mu^2 (mean term), 0.5 s2
    mean_term = 0.5 * (mu**2).mean(0) / LOG2; prec_term = 0.5 * (s2 - np.log(s2) - 1).mean(0) / LOG2
    # Gaussian-channel estimate of I(x;z) per dim: 0.5 log2(1 + var(mu)/mean s2)  (aggregate posterior ~ Gaussian)
    snr = mu.var(0) / s2.mean(0); I_gauss_diag = 0.5 * np.log2(1 + snr)
    C = np.cov(mu.T); Sg = np.diag(s2.mean(0))
    I_gauss_full = 0.5 * (np.linalg.slogdet(C + Sg)[1] - np.linalg.slogdet(Sg)[1]) / LOG2
    # Monte-Carlo I(x;z) under the empirical mixture (saturates at log2 n)
    n = len(mu); idx = rng.choice(n, 400, replace=False); vals = []
    for i in idx:
        z = mu[i] + np.sqrt(s2[i]) * rng.standard_normal(32)
        lq = -0.5 * (((z - mu)**2) / s2 + np.log(2 * np.pi * s2)).sum(1)
        m = lq.max(); vals.append(lq[i] - (m + np.log(np.exp(lq - m).mean())))
    I_mc = float(np.mean(vals) / LOG2)
    # training log
    last = None
    with open(os.path.join(root, run, "step_2500", "metrics.jsonl")) as f:
        tl = [json.loads(l) for l in f if l.strip()]
    trn = [t for t in tl if "kl" in t and not t.get("eval")]
    tail = trn[-10:]
    r = dict(run=run, N=N, K=K, seed=seed, variant=var, n_lj=int(lj.sum()), n_lib=int(lib.sum()),
             rate_total=float(per_dim.sum()), rate_lj=float(kl[lj].sum(1).mean()), rate_lib=float(kl[lib].sum(1).mean()),
             rate_sd_utt=float(kl[tr].sum(1).std()), per_dim=per_dim.tolist(), dim_min=float(per_dim.min()), dim_max=float(per_dim.max()),
             active_001=int((per_dim * LOG2 > 0.01).sum()), active_var=int((mu.var(0) > 0.01).sum()),
             active_1bit=int((per_dim > 1).sum()), pr_rate=pr(per_dim),
             mean_term=float(mean_term.sum()), prec_term=float(prec_term.sum()),
             sigma_mean=float(np.sqrt(s2.mean())), mu_sd=float(np.sqrt(mu.var(0).mean())),
             I_gauss_diag=float(I_gauss_diag.sum()), I_gauss_full=float(I_gauss_full), I_mc=I_mc, log2n=float(np.log2(n)),
             kl_log_last=float(np.mean([t["kl"] for t in tail])), klw=float(tail[-1]["kl_weight"]), recon=float(np.mean([t["recon"] for t in tail])),
             kl_refs=float(kl[~tr].sum(1).mean()))
    res[run] = r
    rows.append([run, r["n_lj"], r["n_lib"], f"{r['rate_total']:.1f}", f"{r['rate_lj']:.1f}", f"{r['rate_lib']:.1f}", f"{r['kl_refs']:.1f}",
                 f"{r['dim_min']:.2f}–{r['dim_max']:.2f}", r["active_001"], r["active_var"], f"{r['sigma_mean']:.3f}", f"{r['mu_sd']:.3f}",
                 f"{r['I_gauss_full']:.1f}", f"{r['I_mc']:.2f}/{r['log2n']:.2f}", f"{r['kl_log_last']:.3f}", f"{r['klw']:.4f}"])
hdr = ["run", "n LJ", "n Libri", "total rate (bits)", "LJ rate", "Libri rate", "refs rate", "bits/dim min–max", "dims KL>0.01 nat", "dims var(mu)>0.01",
       "mean sigma", "mean sd(mu)", "I(x;z) Gauss.", "I(x;z) MC / log2 n", "logged kl (last 10)", "KL weight"]
open(os.path.join(out, "rate_table.md"), "w").write(md_table(hdr, rows))
json.dump(res, open(os.path.join(out, "rate.json"), "w"), indent=1)
print(md_table(hdr, rows))
