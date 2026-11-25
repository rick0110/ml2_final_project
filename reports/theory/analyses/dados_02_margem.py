"""Task 2: variance components of the continuous margin, relative precision margin vs binary, laws for the mean margin."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import norm, chi2
from dados_comum import load, main_cond
from dados_modelos import *
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 60)
df = load(sys.argv[1]); out = sys.argv[2]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
d = main_cond(df)
rng = np.random.default_rng(20261116)

def cube(dd, col):
    """Y[ref, sent] of one run/group (refs sorted)."""
    return dd.pivot(index="reference", columns="sent", values=col).sort_index().values

def anova2(Y):
    a, b = Y.shape
    g = Y.mean(); r = Y.mean(1); c = Y.mean(0)
    msA = b * ((r - g) ** 2).sum() / (a - 1); msB = a * ((c - g) ** 2).sum() / (b - 1)
    E = Y - r[:, None] - c[None, :] + g
    msE = (E ** 2).sum() / ((a - 1) * (b - 1))
    return dict(mean=g, s2_spk=(msA - msE) / b, s2_sent=(msB - msE) / a, s2_res=msE, msA=msA, msB=msB)

# ---------- A. decomposition per run ----------
rows = []
for (run, grp), dd in d.groupby(["run", "group"]):
    for col in ["m", "copied", "ms"]:
        if dd[col].isna().any(): continue
        Y = cube(dd, col); r = anova2(Y); a, b = Y.shape
        tot = max(r["s2_spk"], 0) + max(r["s2_sent"], 0) + r["s2_res"]
        rows.append(dict(run=run, variant=dd.variant.iloc[0], N=dd.N.iloc[0], K=dd.K.iloc[0], seed=dd.seed.iloc[0], group=grp, var=col,
                         n_ref=a, mean=r["mean"], sd_spk=np.sqrt(max(r["s2_spk"], 0)), sd_sent=np.sqrt(max(r["s2_sent"], 0)),
                         sd_res=np.sqrt(r["s2_res"]), icc_spk=max(r["s2_spk"], 0) / tot,
                         se_mean=np.sqrt(max(r["s2_spk"], 0) / a + max(r["s2_sent"], 0) / b + r["s2_res"] / (a * b)),
                         se_iid=Y.std(ddof=1) / np.sqrt(a * b)))
vc = pd.DataFrame(rows); vc.to_csv(os.path.join(out, "varcomp_por_treino.csv"), index=False)
print("=== A. components per run (margin m), plain+wide ===")
print(vc[(vc["var"] == "m") & vc.variant.isin(["plain", "wide"])].drop(columns=["var"]).round(4).to_string())
print("\nmean (RMS) of the components per group, plain runs, variables m and copied:")
P = vc[vc.variant.isin(["plain", "wide"])]
print(P.groupby(["variant", "group", "var"])[["sd_spk", "sd_sent", "sd_res", "icc_spk", "se_mean", "se_iid"]].agg(lambda x: np.sqrt(np.mean(x ** 2))).round(4).to_string())

# ---------- B. joint crossed model run x speaker x utterance (unseen, plain; balanced) ----------
def anova3(Y):
    a, b, c = Y.shape; g = Y.mean()
    A = Y.mean((1, 2)); Bm = Y.mean((0, 2)); C = Y.mean((0, 1))
    AB_ = Y.mean(2); AC = Y.mean(1); BC = Y.mean(0)
    msA = b * c * ((A - g) ** 2).sum() / (a - 1); msB = a * c * ((Bm - g) ** 2).sum() / (b - 1); msC = a * b * ((C - g) ** 2).sum() / (c - 1)
    iab = AB_ - A[:, None] - Bm[None, :] + g; iac = AC - A[:, None] - C[None, :] + g; ibc = BC - Bm[:, None] - C[None, :] + g
    msAB = c * (iab ** 2).sum() / ((a - 1) * (b - 1)); msAC = b * (iac ** 2).sum() / ((a - 1) * (c - 1)); msBC = a * (ibc ** 2).sum() / ((b - 1) * (c - 1))
    E = Y - AB_[:, :, None] - AC[:, None, :] - BC[None, :, :] + A[:, None, None] + Bm[None, :, None] + C[None, None, :] - g
    msE = (E ** 2).sum() / ((a - 1) * (b - 1) * (c - 1))
    return dict(run_x_spk=(msAB - msE) / c, run_x_sent=(msAC - msE) / b, spk_x_sent=(msBC - msE) / a,
                spk=(msB - msAB - msBC + msE) / (a * c), sent=(msC - msAC - msBC + msE) / (a * b), res=msE,
                run_total=(msA - msAB - msAC + msE) / (b * c))
def stack(grp, col, runs):
    return np.stack([cube(d[(d.run == r) & (d.group == grp)], col) for r in runs])
plain = d[d.variant == "plain"]
runs = sorted(plain.run.unique(), key=lambda r: (plain[plain.run == r].N.iloc[0], plain[plain.run == r].K.iloc[0], r))
meta = plain.groupby("run")[["N", "K", "seed"]].first().loc[runs].reset_index()
print("\n=== B. 3-way crossed ANOVA, unseen, 16 plain runs x 4 speakers x 25 utterances (standard deviations) ===")
res3 = {}
for col in ["m", "copied"]:
    Y = stack("unseen", col, runs); r = anova3(Y); res3[col] = r
    print(col, {k: (round(float(np.sign(v) * np.sqrt(abs(v))), 4)) for k, v in r.items()}, "(run_total includes the fixed effect of N,K)")
runs8 = [r for r in runs if not r.startswith("n2_")]
print("seen (8 common refs, 14 runs without N=2):")
for col in ["m", "copied"]:
    Y = stack("seen", col, runs8); r = anova3(Y); res3["seen_" + col] = r
    print(col, {k: (round(float(np.sign(v) * np.sqrt(abs(v))), 4)) for k, v in r.items()})
Wr = sorted(d[d.variant == "wide"].run.unique())
print("external 38 speakers, 3 wide runs:")
for col in ["m", "copied", "ms", "copied_spk"]:
    Y = stack("external", col, Wr); r = anova3(Y); res3["ext_" + col] = r
    print(col, {k: (round(float(np.sign(v) * np.sqrt(abs(v))), 4)) for k, v in r.items()})

# ---------- B2. between-seed variance (6 pairs) and re-synthesis (3 plain/wide pairs) ----------
def pair_stats(r1, r2, grp, col):
    Y1 = cube(d[(d.run == r1) & (d.group == grp)], col); Y2 = cube(d[(d.run == r2) & (d.group == grp)], col)
    D = (Y1 - Y2) / np.sqrt(2); r = anova2(D)
    return dict(pair=f"{r1}|{r2}", group=grp, var=col, mean1=Y1.mean(), mean2=Y2.mean(), diff=Y1.mean() - Y2.mean(),
                half_d2=D.mean() ** 2, s2_x_spk=r["s2_spk"], s2_x_sent=r["s2_sent"], s2_res=r["s2_res"], nref=D.shape[0], nsent=D.shape[1])
pairs = [(a, a.replace("_s1", "_s2")) for a in runs if a.endswith("_s1") and a.replace("_s1", "_s2") in runs]
rs = [pair_stats(a, b, g, c) for a, b in pairs for g in ["unseen", "seen"] for c in ["m", "copied"]]
rs += [pair_stats("n128_k32_s1", "n128_k32_s2", "external", c) for c in ["m", "copied"]]
ps = pd.DataFrame(rs)
print("\n=== B2. seed pairs ==="); print(ps.round(5).to_string())
seedtab = {}
for (g, c), x in ps.groupby(["group", "var"]):
    s2tot = x.half_d2.mean(); n = len(x)
    lo, hi = np.sqrt(n * s2tot / chi2.ppf(.975, n)), np.sqrt(n * s2tot / chi2.ppf(.025, n))
    meas = (x.s2_x_spk.clip(lower=0) / x.nref + x.s2_x_sent.clip(lower=0) / x.nsent + x.s2_res / (x.nref * x.nsent)).mean()
    seedtab[(g, c)] = dict(n_pairs=n, sd_seed_total=np.sqrt(s2tot), ci_lo=lo, ci_hi=hi,
                           sd_seed_x_spk=np.sqrt(max(x.s2_x_spk.mean(), 0)), sd_seed_x_sent=np.sqrt(max(x.s2_x_sent.mean(), 0)), sd_res=np.sqrt(x.s2_res.mean()),
                           sd_from_interactions=np.sqrt(meas), sd_seed_main=np.sqrt(max(s2tot - meas, 0)))
st = pd.DataFrame(seedtab).T; print("\nbetween-seed SD of the run mean (total = what is seen in the metric with the evaluation set fixed), 95% chi-square CI:")
print(st.round(4).to_string())
rp = [pair_stats(w.replace("-wide", ""), w, g, c) for w in Wr for g in ["unseen", "seen"] for c in ["m", "copied"]]
rp = pd.DataFrame(rp); print("\n=== re-synthesis of the same checkpoint (plain vs wide) ==="); print(rp.round(5).to_string())
print(rp.groupby(["group", "var"]).half_d2.agg(lambda x: np.sqrt(x.mean())).round(4).to_string(), "<- re-synthesis SD of the mean")

# ---------- C. precision: margin vs binary ----------
print("\n=== C. relative precision margin vs binary ===")
T = d[d.variant.isin(["plain", "wide"])].groupby(["variant", "run", "N", "K", "seed", "group"]).agg(p=("copied", "mean"), mu=("m", "mean"), sd=("m", "std")).reset_index()
T["z"] = T.mu / T.sd
rows = []
for (run, grp), dd in d[d.variant.isin(["plain", "wide"])].groupby(["run", "group"]):
    Ym = cube(dd, "m"); Yc = cube(dd, "copied"); a, b = Ym.shape
    bs = np.empty((1000, 3)); 
    for i in range(1000):
        si = rng.integers(0, a, a); ti = rng.integers(0, b, b)
        ym = Ym[np.ix_(si, ti)]; bs[i] = [Yc[np.ix_(si, ti)].mean(), ym.mean(), norm.cdf(ym.mean() / ym.std(ddof=1))]
    mu, sd = Ym.mean(), Ym.std(ddof=1); slope = norm.pdf(mu / sd) / sd
    rows.append(dict(run=run, group=grp, N=dd.N.iloc[0], K=dd.K.iloc[0], p=Yc.mean(), mu=mu, sd=sd, se_p=bs[:, 0].std(), se_mu=bs[:, 1].std(),
                     se_phi=bs[:, 2].std(), slope=slope, se_mu_in_p=bs[:, 1].std() * slope))
pr = pd.DataFrame(rows); pr["eff_var_ratio"] = (pr.se_p / pr.se_phi) ** 2
pr.to_csv(os.path.join(out, "precisao_por_treino.csv"), index=False)
print("speaker x utterance bootstrap per run; se_phi = SE of Phi(mean/sd) (margin-based estimator of p); eff = (se_p/se_phi)^2")
print(pr.round(4).to_string())
print(pr.groupby("group")[["se_p", "se_phi", "eff_var_ratio"]].median().round(4))
# common scale via the empirical slope p ~ mu across runs
for grp in ["unseen", "seen"]:
    t = T[(T.group == grp) & (T.variant == "plain")]
    sl = np.polyfit(t.mu, t.p, 1)[0]; r = np.corrcoef(t.mu, t.p)[0, 1]
    sp = st.loc[(grp, "copied"), "sd_seed_total"]; sm = st.loc[(grp, "m"), "sd_seed_total"]
    rng_p = t.groupby(["N", "K"]).p.mean(); rng_m = t.groupby(["N", "K"]).mu.mean()
    print(f"{grp}: slope dp/dmu across runs={sl:.2f}, r={r:.3f}; seed SD p={sp:.4f}, margin={sm:.4f} -> in units of p {sm*sl:.4f}; "
          f"variance ratio (binary/margin)={(sp/(sm*sl))**2:.2f}; signal/noise: SD between configs / seed SD: p {rng_p.std()/sp:.2f}, margin {rng_m.std()/sm:.2f}")
    rr = rp[(rp.group == grp)]
    a_ = np.sqrt(rr[rr["var"] == "copied"].half_d2.mean()); b_ = np.sqrt(rr[rr["var"] == "m"].half_d2.mean())
    print(f"   re-synthesis: SD p={a_:.4f}, margin={b_:.4f} -> in p {b_*sl:.4f}; var ratio={(a_/(b_*sl))**2:.2f}")

# ---------- D. laws for the mean margin ----------
def fit_table(models, N, K, y, cfg, label):
    t = pd.DataFrame([summary(m, N, K, y, cfg) for m in models])
    t["dAICc"] = t.aicc - t.aicc.min(); t["dBIC"] = t.bic - t.bic.min()
    w = np.exp(-0.5 * t.dAICc); t["w_AICc"] = w / w.sum()
    print(f"\n--- {label} (n={len(y)} runs, {len(set(cfg))} configurations) ---")
    print(t[["model", "k", "rmse", "cv_rmse_run", "cv_rmse_cfg", "dAICc", "dBIC", "w_AICc", "par"]].assign(par=t.par.map(lambda p: np.round(p, 4))).round(4).to_string())
    return t
N = meta.N.values.astype(float); K = meta.K.values.astype(float); cfg, cfgn = config_ids(N, K)
Ym = stack("unseen", "m", runs); y = Ym.mean((1, 2))
print("\n=== D. laws for the mean margin, unseen ===")
MM = margin_models()
t_all = fit_table(MM, N, K, y, cfg, "unseen, mean margin, all")
keep = N > 2
cfg2, _ = config_ids(N[keep], K[keep])
t_no2 = fit_table(MM, N[keep], K[keep], y[keep], cfg2, "unseen, mean margin, without N=2")
t_all.drop(columns="par").to_csv(os.path.join(out, "leis_margem_unseen.csv"), index=False)
sd_seed_m = st.loc[("unseen", "m"), "sd_seed_total"]
print(f"\nreference: between-seed SD of the mean margin = {sd_seed_m:.4f} (floor of the RMSE attainable per run)")

# speaker x utterance bootstrap: CV-RMSE differences and parameters
names = [m.name for m in MM]
def cvvec(yb, Nn, Kk, cf):
    out_ = []
    for m in MM:
        e = loco(m, Nn, Kk, yb, cf) - yb
        out_.append(np.sqrt(np.mean(e ** 2)))
    return np.array(out_)
cv0 = cvvec(y, N, K, cfg)
cvb = np.empty((B, len(MM))); alph = np.empty(B); db = np.empty((B, len(y)))
ia = names.index("a + b N^-alpha + c ln K")
for i in range(B):
    si = rng.integers(0, 4, 4); ti = rng.integers(0, 25, 25)
    yb = Ym[:, si][:, :, ti].mean((1, 2)); db[i] = yb - y
    cvb[i] = cvvec(yb, N, K, cfg); alph[i] = MM[ia].fit(N, K, yb)[2]
np.save(os.path.join(out, "boot_dev_margem_unseen.npy"), db)
print("\nBootstrap (speakers x utterances, B=%d) of the CV-RMSE (leave-one-config-out) per law, unseen margin:" % B)
bt = pd.DataFrame(dict(model=names, cv=cv0, lo=np.percentile(cvb, 2.5, 0), med=np.median(cvb, 0), hi=np.percentile(cvb, 97.5, 0),
                       p_best=np.bincount(cvb.argmin(1), minlength=len(MM)) / B)); print(bt.round(4).to_string())
ref = names.index("a + b ln N + c ln K")
print("CV-RMSE difference relative to 'a + b ln N + c ln K' (positive = worse): median [2.5, 97.5], P(diff>0)")
for j, nm in enumerate(names):
    dd_ = cvb[:, j] - cvb[:, ref]; print(f"  {nm:38s} {np.median(dd_):+.4f} [{np.percentile(dd_,2.5):+.4f}, {np.percentile(dd_,97.5):+.4f}]  P={np.mean(dd_>0):.2f}")
print(f"exponent alpha of 'a + b N^-alpha + c ln K': estimate {MM[ia].fit(N,K,y)[2]:.2f}, speaker/utterance bootstrap 95% CI [{np.percentile(alph,2.5):.2f}, {np.percentile(alph,97.5):.2f}] (grid 0.02..2)")

# parametric bootstrap with seed noise: confusion matrix (generate from law i, choose by CV and by AICc)
print("\nIdentifiability with seed noise (parametric; y_sim = fit of the generating law + N(0, sd_seed^2) per run, B=%d):" % min(B, 1000))
sel = [0, 1, 2, 3, 4, 5, 6, 7]
conf_cv = np.zeros((len(sel), len(sel))); conf_aic = np.zeros_like(conf_cv); alpha_sim = {}
for a_, i in enumerate(sel):
    par = MM[i].fit(N, K, y); yh = MM[i].predict(par, N, K); al = []
    for b_ in range(min(B, 1000)):
        ys = yh + rng.normal(0, sd_seed_m, len(y))
        s = [summary(MM[j], N, K, ys, cfg) for j in sel]
        conf_cv[a_, int(np.argmin([x["cv_rmse_run"] for x in s]))] += 1; conf_aic[a_, int(np.argmin([x["aicc"] for x in s]))] += 1
        al.append(s[sel.index(ia)]["par"][-1])
    alpha_sim[names[i]] = np.percentile(al, [2.5, 50, 97.5])
short = ["1/N", "L1", "1/rtN", "L2", "lnN", "lnN+lnK", "pow", "pow+lnK"]
print("rows = generating law; columns = law chosen by lowest CV-RMSE (fraction)")
print(pd.DataFrame(conf_cv / conf_cv.sum(1, keepdims=True), index=short, columns=short).round(2).to_string())
print("same, chosen by AICc")
print(pd.DataFrame(conf_aic / conf_aic.sum(1, keepdims=True), index=short, columns=short).round(2).to_string())
print("estimated alpha (law pow+lnK) when the truth is each law: percentiles 2.5/50/97.5"); [print(f"  {k:38s}", np.round(v, 2)) for k, v in alpha_sim.items()]

# seen: mean margin
print("\n=== D2. mean margin, seen (8 refs; N=2 has 2 speakers x 4 refs) ===")
ys = np.array([d[(d.run == r) & (d.group == "seen")].m.mean() for r in runs])
MS = [Model("a + c/K", 2, "lin", lambda N, K: np.c_[one(N), 1 / K]), Model("a + c/K + b/N", 3, "lin", lambda N, K: np.c_[one(N), 1 / K, 1 / N]),
      Model("a + c/sqrt(K)", 2, "lin", lambda N, K: np.c_[one(N), K ** -.5]), Model("a + c/sqrt(K) + b/sqrt(N)", 3, "lin", lambda N, K: np.c_[one(N), K ** -.5, N ** -.5]),
      Model("a + c ln K", 2, "lin", lambda N, K: np.c_[one(N), np.log(K)]), Model("a + c ln K + b ln N", 3, "lin", lambda N, K: np.c_[one(N), np.log(K), np.log(N)]),
      Model("a + c K^-beta", 3, "sep", lambda N, K, th: np.c_[one(N), K ** -th], ALPHAS),
      Model("a + c K^-beta + b ln N", 4, "sep", lambda N, K, th: np.c_[one(N), K ** -th, np.log(N)], ALPHAS)]
fit_table(MS, N, K, ys, cfg, "seen, mean margin, all")
fit_table(MS, N[keep], K[keep], ys[keep], cfg2, "seen, mean margin, without N=2")
print("seed SD of the seen margin:", round(st.loc[("seen", "m"), "sd_seed_total"], 4))
json.dump({f"{g}|{c}": v for (g, c), v in seedtab.items()}, open(os.path.join(out, "seed_sd.json"), "w"), indent=1)
