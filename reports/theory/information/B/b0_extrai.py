"""b0: extracts, for each training run, the whitened latent geometry (trained voices, LJ default voice, references)
and the per-reference statistics (copy, margin, WER, duration) at R=1, R=4 and strength 0.6.
Usage: python3 -I b0_extrai.py <latents_root> <results_root> <out.npz>
Data are only read; writes only <out.npz> (pickle disabled)."""
import sys, os, re, csv, json
import numpy as np

lat_root, res_root, out = sys.argv[1], sys.argv[2], sys.argv[3]

def parse_run(name):
    m = re.match(r"n(\d+)_k(\d+)_s(\d+)(?:-(.*))?$", name)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4) or "") if m else None

def fnum(x):
    try: return float(x)
    except Exception: return np.nan

def read_res(run, sub):
    p = os.path.join(res_root, run, "step_2500", sub, "results.csv")
    if not os.path.exists(p): return None
    by = {}
    with open(p, encoding="utf-8") as f:
        for x in csv.DictReader(f):
            by.setdefault(x["reference"], []).append(x)
    return by

def summ(rows):
    sr = np.array([fnum(x["sim_ref"]) for x in rows]); sd = np.array([fnum(x["sim_default"]) for x in rows])
    ss = np.array([fnum(x.get("sim_spk", "nan")) for x in rows]); w = np.array([fnum(x["wer"]) for x in rows])
    du = np.array([fnum(x["duration_s"]) for x in rows])
    m = sr - sd
    return dict(n=len(rows), k=int((sr > sd).sum()), k_spk=int(np.nansum(ss > sd)) if np.isfinite(ss).any() else -1,
                m=float(m.mean()), m_sd=float(m.std(ddof=1)), ms=float(np.nanmean(ss - sd)) if np.isfinite(ss).any() else np.nan,
                sim_ref=float(sr.mean()), sim_def=float(sd.mean()), wer=float(np.nanmean(w)),
                f0=fnum(rows[0]["ref_f0_hz"]), dur=float(np.nanmean(du)),
                per_m=m.tolist(), per_dur=du.tolist(), per_wer=w.tolist(), per_copy=(sr > sd).astype(int).tolist())

runs = sorted(r for r in os.listdir(lat_root) if parse_run(r) and os.path.exists(os.path.join(lat_root, r, "step_2500", "latents.npz")))
store = {}; meta = []
for run in runs:
    N, K, seed, var = parse_run(run)
    z = np.load(os.path.join(lat_root, run, "step_2500", "latents.npz"), allow_pickle=False)
    mu = z["mu"].astype(np.float64); spk = z["speaker"].astype(str); src = z["source"].astype(str)
    emo = z["emotion"].astype(str); uid = z["utt_id"].astype(str); lv = z["logvar"].astype(np.float64)
    tr = src != "refs"; lj = tr & (spk == "lj"); lib = tr & ~lj
    # within-speaker covariance from LibriTTS training latents
    labs = spk[lib]; X = mu[lib]
    u, inv, cnt = np.unique(labs, return_inverse=True, return_counts=True)
    Mn = np.zeros((len(u), 32)); np.add.at(Mn, inv, X); Mn /= cnt[:, None]
    R = X - Mn[inv]; W = R.T @ R / (len(X) - len(u))
    L = np.linalg.cholesky(W); Li = np.linalg.inv(L)
    wh = lambda A: A @ Li.T
    # posterior noise (VAE sigma^2) in whitened units: mean over directions of diag(Li diag(s2) Li^T)
    s2 = np.exp(lv[lib]).mean(0); post_w = float(np.trace(Li @ np.diag(s2) @ Li.T) / 32)
    ljw = wh(mu[lj]); v0 = ljw.mean(0); lj_within = float(((ljw - v0) ** 2).sum(1).mean())
    # between-speaker covariance (whitened), unbiased
    n0 = (len(X) - (cnt ** 2).sum() / len(X)) / (len(u) - 1)
    D = Mn - X.mean(0); SSB = (D * cnt[:, None]).T @ D
    Sb = (SSB / (len(u) - 1) - W) / n0; Sbw = Li @ Sb @ Li.T
    refs = np.where(src == "refs")[0]
    res1 = read_res(run, "strength_1.0"); res4 = read_res(run, "strength_1.0_r4"); res6 = read_res(run, "strength_0.6")
    R_ = []
    for i in refs:
        rid = uid[i]
        if res1 is None or rid not in res1: continue
        rec = dict(ref=rid, grp=emo[i], spk=spk[i], c=wh(mu[i:i+1])[0].tolist(), c_raw=mu[i].tolist(), s2_ref=np.exp(lv[i]).tolist(), kl=float(0.5 * (mu[i]**2 + np.exp(lv[i]) - lv[i] - 1).sum() / np.log(2)))
        rec["r1"] = summ(res1[rid])
        if res4 is not None and rid in res4: rec["r4"] = summ(res4[rid])
        if res6 is not None and rid in res6: rec["s06"] = summ(res6[rid])
        # is the reference utterance among the training latents? (exclude from speaker means if so)
        rec["in_train"] = bool(np.any(uid[tr] == rid))
        R_.append(rec)
    # trained speaker means excluding reference utterances
    excl = set(r["ref"] for r in R_)
    keep = lib & ~np.isin(uid, list(excl))
    u2, inv2, cnt2 = np.unique(spk[keep], return_inverse=True, return_counts=True)
    M2 = np.zeros((len(u2), 32)); np.add.at(M2, inv2, wh(mu[keep])); M2 /= cnt2[:, None]
    store[run] = dict(N=N, K=K, seed=seed, var=var, spk_ids=u2.tolist(), spk_means=M2.tolist(), spk_n=cnt2.tolist(),
                      v0=v0.tolist(), n_lj=int(lj.sum()), lj_within=lj_within, post_w=post_w, Sbw=Sbw.tolist(),
                      refs=R_, raw=dict(W=W.tolist(), s2_lib=s2.tolist(), s2_lj=np.exp(lv[lj]).mean(0).tolist(),
                      W_lj=np.cov(mu[lj].T).tolist(), m_lj=mu[lj].mean(0).tolist(),
                      means=np.array([mu[keep & (spk == s_)].mean(0) for s_ in u2]).tolist(), Sb=Sb.tolist()))
    meta.append((run, N, K, seed, var, len(u2), int(lj.sum()), len(R_), post_w, lj_within))
    print(run, N, K, seed, var, "S=", len(u2), "nLJ=", int(lj.sum()), "refs=", len(R_), "post_w=%.3f" % post_w, "LJ within=%.1f" % lj_within,
          "inTrain=", sum(r["in_train"] for r in R_), flush=True)
np.savez_compressed(out, data=np.array(json.dumps(store)))
