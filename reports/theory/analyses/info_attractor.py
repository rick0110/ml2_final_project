"""Tasks 4-5: latent geometry of the 38 external references vs per-speaker copy success in the -wide runs.
usage: python3 -I info_attractor.py DATA OUT [nboot]"""
import sys, os, json, warnings; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import spearmanr, rankdata
from scipy.optimize import minimize
from info_common import *
from info_speaker_lib import gauss_estimates, scatter
warnings.filterwarnings("ignore")
root, out = sys.argv[1], sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
rng = np.random.default_rng(2)
RUNS = ["n8_k32_s1-wide", "n32_k32_s1-wide", "n128_k32_s1-wide"]

def fnum(x):
    try: return float(x)
    except Exception: return np.nan

def per_speaker(rows, use_spk=False):
    by = {}
    for x in rows:
        if x["reference_emotion"] != "external": continue
        sr, sd = fnum(x["sim_spk"] if use_spk else x["sim_ref"]), fnum(x["sim_default"])
        by.setdefault(x["reference_speaker"], []).append((sr > sd, sr - sd, sr, sd, fnum(x["wer"]), fnum(x["ref_f0_hz"]), x["reference"]))
    return by

def fit_logit(X, k, n, l2=1e-3):
    """binomial logistic regression, k successes of n; X includes intercept column"""
    def nll(b):
        e = X @ b; return -(k * e - n * np.logaddexp(0, e)).sum() + l2 * (b[1:] ** 2).sum()
    return minimize(nll, np.zeros(X.shape[1]), method="BFGS").x

def loo_bits(X, k, n):
    """leave-one-speaker-out predictive log-loss per sample (bits); lower is better"""
    tot = 0.0
    for i in range(len(k)):
        m = np.ones(len(k), bool); m[i] = False
        b = fit_logit(X[m], k[m], n[m]); e = X[i] @ b
        p = 1 / (1 + np.exp(-e)); p = np.clip(p, 1e-4, 1 - 1e-4)
        tot += -(k[i] * np.log2(p) + (n[i] - k[i]) * np.log2(1 - p))
    return tot / n.sum()

def zs(v): return (v - np.nanmean(v)) / np.nanstd(v)

res = {}; pooled = []
for run in RUNS:
    d = load_latents(root, run); tr = d["train"]; spk = d["speaker"]
    lj = tr & (spk == "lj"); lib = tr & ~lj
    mu = d["mu"]; s2 = np.exp(d["logvar"])
    X, lab = mu[lib], spk[lib]
    u, cnt = np.unique(lab, return_counts=True); keep = np.isin(lab, u[cnt >= 2])
    g = gauss_estimates(X[keep], lab[keep]); W = g["W"]; Linv = np.linalg.inv(np.linalg.cholesky(W))
    uu, inv, cc, M, _, _ = scatter(X, lab)              # trained Libri speaker means (all speakers present)
    m_lj = mu[lj].mean(0)
    sd_lj_within = np.sqrt(np.trace(np.cov(mu[lj].T)))
    refs = (d["source"] == "refs") & (d["emotion"] == "external")
    ref_id = {d["utt_id"][i]: i for i in np.where(refs)[0]}
    rows = load_results(root, run); by = per_speaker(rows)
    rows4 = load_results(root, run, "strength_1.0_r4"); by4 = per_speaker(rows4, use_spk=True)
    # LJ-vs-Libri discriminant direction (whitened)
    dvec = Linv @ (m_lj - X.mean(0)); dvec /= np.linalg.norm(dvec)
    recs = []
    for s, v in sorted(by.items()):
        rid = v[0][6]
        if rid not in ref_id: continue
        z = mu[ref_id[rid]]; zw = Linv @ z
        dl = np.linalg.norm(z - m_lj); dlw = np.linalg.norm(zw - Linv @ m_lj)
        dn_all = np.linalg.norm(M - z, axis=1); dnw_all = np.linalg.norm((M - z) @ Linv.T, axis=1)
        a = np.array(v, dtype=object)
        rec = dict(spk=s, n=len(v), k=int(sum(x[0] for x in v)), margin=float(np.nanmean([x[1] for x in v])),
                   sim_ref=float(np.nanmean([x[2] for x in v])), sim_def=float(np.nanmean([x[3] for x in v])),
                   wer=float(np.nanmean([x[4] for x in v])), f0=float(v[0][5]),
                   d_lj=float(dl), d_nn=float(dn_all.min()), d_lj_w=float(dlw), d_nn_w=float(dnw_all.min()),
                   proj_lj=float((zw - Linv @ X.mean(0)) @ dvec),
                   kl=float(kl_bits(z, d["logvar"][ref_id[rid]]).sum()))
        if s in by4: rec["copy4"] = float(np.mean([x[0] for x in by4[s]]))
        rec["copy"] = rec["k"] / rec["n"]
        recs.append(rec)
    R = {k: np.array([r[k] for r in recs], float) for k in recs[0] if k != "spk"}
    R["logf0"] = np.log(R["f0"]); R["ratio_w"] = R["d_nn_w"] / R["d_lj_w"]
    out_r = dict(run=run, S_ref=len(recs), copy_mean=float(R["copy"].mean()), d_lj_mean=float(R["d_lj"].mean()), d_nn_mean=float(R["d_nn"].mean()),
                 d_lj_w_mean=float(R["d_lj_w"].mean()), d_nn_w_mean=float(R["d_nn_w"].mean()),
                 within_rms_w=float(np.sqrt(32)), lj_within_rms=float(sd_lj_within), ref_kl_mean=float(R["kl"].mean()),
                 lj_center_to_lib_center_w=float(np.linalg.norm(Linv @ (m_lj - X.mean(0)))))
    preds = ["d_lj", "d_nn", "d_lj_w", "d_nn_w", "proj_lj", "logf0", "ratio_w"]
    targets = ["copy", "margin", "sim_ref", "sim_def"]
    rho = {}
    for p_ in preds:
        for t in targets:
            rho[f"{t}~{p_}"] = float(spearmanr(R[p_], R[t]).correlation)
    # bootstrap over speakers
    n_ = len(recs); bs = {k: [] for k in ["copy~d_lj_w", "copy~d_nn_w", "copy~logf0", "diff_lj_nn", "diff_lj_f0", "margin~d_lj_w", "sim_ref~d_lj_w", "sim_def~d_lj_w"]}
    for _ in range(NB):
        i = rng.integers(0, n_, n_)
        a = spearmanr(R["d_lj_w"][i], R["copy"][i]).correlation; b = spearmanr(R["d_nn_w"][i], R["copy"][i]).correlation
        c = spearmanr(R["logf0"][i], R["copy"][i]).correlation
        bs["copy~d_lj_w"].append(a); bs["copy~d_nn_w"].append(b); bs["copy~logf0"].append(c)
        bs["diff_lj_nn"].append(abs(a) - abs(b)); bs["diff_lj_f0"].append(abs(a) - abs(c))
        bs["margin~d_lj_w"].append(spearmanr(R["d_lj_w"][i], R["margin"][i]).correlation)
        bs["sim_ref~d_lj_w"].append(spearmanr(R["d_lj_w"][i], R["sim_ref"][i]).correlation)
        bs["sim_def~d_lj_w"].append(spearmanr(R["d_lj_w"][i], R["sim_def"][i]).correlation)
    ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in bs.items()}
    ci["P(|rho_lj|>|rho_nn|)"] = float(np.mean(np.array(bs["diff_lj_nn"]) > 0)); ci["P(|rho_lj|>|rho_f0|)"] = float(np.mean(np.array(bs["diff_lj_f0"]) > 0))
    # partial Spearman: copy ~ d_lj_w controlling for log f0 (rank residuals)
    def partial(a, b, c):
        ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
        ea = ra - np.polyval(np.polyfit(rc, ra, 1), rc); eb = rb - np.polyval(np.polyfit(rc, rb, 1), rc)
        return float(np.corrcoef(ea, eb)[0, 1])
    rho["copy~d_lj_w|f0"] = partial(R["d_lj_w"], R["copy"], R["logf0"]); rho["copy~f0|d_lj_w"] = partial(R["logf0"], R["copy"], R["d_lj_w"])
    rho["copy~d_nn_w|d_lj_w"] = partial(R["d_nn_w"], R["copy"], R["d_lj_w"]); rho["d_lj_w~logf0"] = float(spearmanr(R["d_lj_w"], R["logf0"]).correlation)
    rho["d_lj_w~d_nn_w"] = float(spearmanr(R["d_lj_w"], R["d_nn_w"]).correlation)
    # logistic models, LOO over speakers
    k, nn = R["k"], R["n"]; one = np.ones(n_)
    models = {"intercepto": [], "d_lj": ["d_lj_w"], "d_nn": ["d_nn_w"], "f0": ["logf0"], "d_lj+d_nn": ["d_lj_w", "d_nn_w"],
              "d_lj+f0": ["d_lj_w", "logf0"], "d_lj+d_nn+f0": ["d_lj_w", "d_nn_w", "logf0"]}
    lb = {}
    for name, cols in models.items():
        Xm = np.column_stack([one] + [zs(R[c]) for c in cols]); lb[name] = loo_bits(Xm, k, nn)
    # threshold along d_lj (raw whitened units): P(copy)=0.5 at d50, transition width 1/slope
    b = fit_logit(np.column_stack([one, R["d_lj_w"]]), k, nn, l2=0)
    d50 = -b[0] / b[1]; width = 1 / b[1]
    bsd = []
    for _ in range(500):
        i = rng.integers(0, n_, n_); bb = fit_logit(np.column_stack([one, R["d_lj_w"][i]]), k[i], nn[i], l2=0)
        if bb[1] > 0: bsd.append((-bb[0] / bb[1], 1 / bb[1]))
    bsd = np.array(bsd)
    # overdispersion: per-speaker copy is nearly all-or-nothing
    p_hat = R["copy"].mean(); disp = float(R["copy"].var() / (p_hat * (1 - p_hat) / 25))
    out_r.update(rho=rho, ci=ci, loo_bits=lb, d50_w=float(d50), width_w=float(width),
                 d50_ci=[float(np.percentile(bsd[:, 0], 2.5)), float(np.percentile(bsd[:, 0], 97.5))],
                 width_ci=[float(np.percentile(bsd[:, 1], 2.5)), float(np.percentile(bsd[:, 1], 97.5))],
                 overdispersion=disp, frac_all_or_none=float(np.mean((R["copy"] <= 0.1) | (R["copy"] >= 0.9))),
                 copy4_mean=float(np.nanmean(R.get("copy4", np.full(n_, np.nan)))),
                 per_speaker=[{k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in r.items()} for r in recs])
    res[run] = out_r
    for r in recs: pooled.append(dict(run=run, **{k_: r[k_] for k_ in ["spk", "k", "n", "copy", "margin", "d_lj_w", "d_nn_w", "f0"]}))
    print(run, json.dumps({k_: v_ for k_, v_ in out_r.items() if k_ not in ("per_speaker",)}, default=float)[:3000], flush=True)

# ---------- pooled over runs: z-score predictors within run, run fixed effects, speaker-cluster bootstrap
spks = sorted(set(p["spk"] for p in pooled))
def design(sel_rows, cols):
    Xs = []
    for run in RUNS:
        rr = [p for p in sel_rows if p["run"] == run]
        if not rr: continue
        Z = {c: zs(np.array([p[c] if c != "f0" else np.log(p[c]) for p in rr], float)) for c in cols}
        for j, p in enumerate(rr):
            Xs.append([1.0 * (run == q) for q in RUNS] + [Z[c][j] for c in cols])
    return np.array(Xs)
pk = np.array([p["k"] for p in pooled], float); pn = np.array([p["n"] for p in pooled], float)
pool = {}
for name, cols in {"run": [], "d_lj": ["d_lj_w"], "d_nn": ["d_nn_w"], "f0": ["f0"], "d_lj+d_nn": ["d_lj_w", "d_nn_w"], "d_lj+d_nn+f0": ["d_lj_w", "d_nn_w", "f0"]}.items():
    Xd = design(pooled, cols); b = fit_logit(Xd, pk, pn, l2=0); pool[name] = dict(coef=b[3:].tolist())
    # leave-one-SPEAKER-out (all runs of that speaker) predictive bits/sample
    tot = 0.0
    for s in spks:
        m = np.array([p["spk"] != s for p in pooled]); bb = fit_logit(Xd[m], pk[m], pn[m])
        e = Xd[~m] @ bb; p_ = np.clip(1 / (1 + np.exp(-e)), 1e-4, 1 - 1e-4)
        tot += -(pk[~m] * np.log2(p_) + (pn[~m] - pk[~m]) * np.log2(1 - p_)).sum()
    pool[name]["loso_bits"] = float(tot / pn.sum())
# cluster bootstrap of the pooled standardized coefficients in the d_lj+d_nn+f0 model
Xd = design(pooled, ["d_lj_w", "d_nn_w", "f0"]); bsc = []
idx_by = {s: [i for i, p in enumerate(pooled) if p["spk"] == s] for s in spks}
for _ in range(1000):
    pick = rng.choice(spks, len(spks)); ii = np.concatenate([idx_by[s] for s in pick])
    bsc.append(fit_logit(Xd[ii], pk[ii], pn[ii], l2=0)[3:])
bsc = np.array(bsc)
pool["coef_ci_d_lj+d_nn+f0"] = [[float(np.percentile(bsc[:, j], 2.5)), float(np.percentile(bsc[:, j], 97.5))] for j in range(3)]
# pooled Spearman (within-run ranks), cluster bootstrap
def pooled_rho(rows, a, b):
    ra, rb = [], []
    for run in RUNS:
        rr = [p for p in rows if p["run"] == run]
        if len(rr) < 3: continue
        ra += list(zs(rankdata([p[a] for p in rr]))); rb += list(zs(rankdata([p[b] for p in rr])))
    return float(np.corrcoef(ra, rb)[0, 1])
pr_ = {}
for a in ["d_lj_w", "d_nn_w", "f0"]:
    v = pooled_rho(pooled, a, "copy"); bb = []
    for _ in range(1000):
        pick = rng.choice(spks, len(spks)); rows = [pooled[i] for s in pick for i in idx_by[s]]; bb.append(pooled_rho(rows, a, "copy"))
    pr_[a] = [v, float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]
pool["spearman_within_run"] = pr_
res["pooled"] = pool
print("pooled", json.dumps(pool, default=float))
json.dump(res, open(os.path.join(out, "attractor.json"), "w"), indent=1, default=float)
