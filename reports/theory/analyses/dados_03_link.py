"""Task 3: does P(copied) = Phi(mean/sd) predict the binary fraction? All-or-none per speaker (ICC).
Usage: python3 -I dados_03_link.py <results_dir> <out_dir>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import norm, binom, spearmanr
from scipy.optimize import minimize_scalar, minimize
from dados_comum import load, main_cond
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
df = load(sys.argv[1]); out = sys.argv[2]
d = main_cond(df)
rng = np.random.default_rng(3)
rows = []
for (run, grp), x in d.groupby(["run", "group"]):
    mu, sd = x.m.mean(), x.m.std(ddof=1)
    # per-speaker mixture: mean over speakers of Phi(mu_s / sd_w), sd_w = within-speaker sd (pooled)
    g = x.groupby("spk").m
    sw = np.sqrt((g.var(ddof=1) * (g.size() - 1)).sum() / (len(x) - g.ngroups))
    pmix = norm.cdf(g.mean() / sw).mean()
    # empirical per speaker: fraction copied per speaker
    ps = x.groupby("spk").copied.mean()
    rows.append(dict(run=run, variant=x.variant.iloc[0], group=grp, N=x.N.iloc[0], K=x.K.iloc[0], n_spk=g.ngroups, p=x.copied.mean(),
                     phi_global=norm.cdf(mu / sd), phi_mix=pmix, mu=mu, sd=sd, sd_within=sw,
                     frac_spk_0=(ps <= 0.1).mean(), frac_spk_1=(ps >= 0.9).mean()))
L = pd.DataFrame(rows); L.to_csv(os.path.join(out, "link_probit.csv"), index=False)
P = L[L.variant.isin(["plain", "wide"])]
print("=== Phi(mean/sd) vs fraction copied (plain + wide) ===")
print(P.round(3).to_string())
for col in ["phi_global", "phi_mix"]:
    e = P[col] - P.p
    print(f"{col}: mean bias {e.mean():+.4f}, RMSE {np.sqrt((e**2).mean()):.4f}, corr {np.corrcoef(P[col], P.p)[0,1]:.3f}, max|e| {e.abs().max():.3f}")
    for g_, x in P.groupby("group"):
        e = x[col] - x.p; print(f"   {g_:9s} n={len(x)} bias {e.mean():+.4f} RMSE {np.sqrt((e**2).mean()):.4f}")
# error expected from cluster-binomial sampling alone: SE of p within the run (speaker x utterance bootstrap already in precisao_por_treino)
pr = pd.read_csv(os.path.join(out, "precisao_por_treino.csv"))
M = P.merge(pr[["run", "group", "se_p"]], on=["run", "group"])
M["z_global"] = (M.phi_global - M.p) / M.se_p
print("deviation Phi_global - p in units of the bootstrap SE of p: median |z| %.2f, max %.2f" % (M.z_global.abs().median(), M.z_global.abs().max()))

print("\n=== All-or-none per speaker ===")
def overdisp(x):
    """x: samples of one run/group. Overdispersion test of the per-speaker fraction vs a homogeneous binomial,
    and ICC on the latent scale (probit with a random intercept per speaker, maximum-likelihood fit with quadrature)."""
    k = x.groupby("spk").copied.sum().values; n = x.groupby("spk").copied.size().values
    p = k.sum() / n.sum()
    chi = (((k - n * p) ** 2) / (n * p * (1 - p) + 1e-12)).sum()
    exp_ext = np.mean([binom.cdf(np.floor(0.1 * ni), ni, p) + binom.sf(np.ceil(0.9 * ni) - 1, ni, p) for ni in n])
    obs_ext = np.mean((k / n <= 0.1) | (k / n >= 0.9))
    return p, chi, len(k) - 1, obs_ext, exp_ext
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(40); gh_w = gh_w / gh_w.sum()
def probit_icc(k, n):
    def nll(th):
        b0, ls = th; s = np.exp(np.clip(ls, -6, 3))
        pr_ = np.clip(norm.cdf(b0 + s * gh_x), 1e-12, 1 - 1e-12)[None, :]
        ll = k[:, None] * np.log(pr_) + (n - k)[:, None] * np.log(1 - pr_)
        mx = ll.max(1, keepdims=True)
        return -(mx[:, 0] + np.log((np.exp(ll - mx) * gh_w[None, :]).sum(1))).sum()
    r = minimize(nll, [0.0, 0.0], method="Nelder-Mead")
    s2 = np.exp(2 * np.clip(r.x[1], -6, 3)); return s2 / (s2 + 1), r.x
rows = []
for (run, grp), x in d[d.variant.isin(["plain", "wide"])].groupby(["run", "group"]):
    p, chi, dfree, oe, ee = overdisp(x)
    k = x.groupby("spk").copied.sum().values; n = x.groupby("spk").copied.size().values
    icc_l, th = probit_icc(k, n)
    # ICC of the continuous margin (one-way ANOVA)
    g = x.groupby("spk").m; nb = g.size().mean()
    msb = nb * ((g.mean() - x.m.mean()) ** 2).sum() / (g.ngroups - 1); msw = (g.var(ddof=1) * (g.size() - 1)).sum() / (len(x) - g.ngroups)
    icc_m = max((msb - msw) / nb, 0) / (max((msb - msw) / nb, 0) + msw)
    rows.append(dict(run=run, group=grp, n_spk=len(k), p=p, chi2_overdisp=chi, df=dfree, frac_extremos_obs=oe, frac_extremos_binom=ee,
                     icc_latente_probit=icc_l, icc_margem=icc_m))
O = pd.DataFrame(rows); O.to_csv(os.path.join(out, "tudo_ou_nada.csv"), index=False)
print(O.round(3).to_string())
print(O.groupby("group")[["frac_extremos_obs", "frac_extremos_binom", "icc_latente_probit", "icc_margem"]].median().round(3))
# wide: distribution of the per-speaker rates
W = d[(d.variant == "wide") & (d.group == "external")]
R = W.groupby(["N", "spk"]).copied.mean().unstack(0)
print("\nwide, 38 external speakers: histogram of the per-speaker copy rate (bins 0,.1,..,1)")
for n_ in R.columns:
    h = np.histogram(R[n_], bins=[0, .1001, .3, .5, .7, .8999, 1.0001])[0]; print(f"N={n_}: {h}  (<=10%: {h[0]}, >=90%: {h[-1]})")
print("Spearman between N:", {f"{a}-{b}": round(spearmanr(R[a], R[b])[0], 2) for a, b in [(8, 32), (32, 128), (8, 128)]})
Rm = W.groupby(["N", "spk"]).m.mean().unstack(0)
print("same for the mean margin per speaker:", {f"{a}-{b}": round(spearmanr(Rm[a], Rm[b])[0], 2) for a, b in [(8, 32), (32, 128), (8, 128)]})
