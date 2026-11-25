"""b1 (revision v2): points 1, 3, 4, 5a, 5b, 8, 10 of the critique.
python3 -I b1_revisao.py <coverage2> <results> <out>
"""
import sys, os, json, csv
import numpy as np
from scipy.stats import norm, spearmanr
from scipy.special import logsumexp, expit
from scipy.optimize import brentq, minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_lib import load_latents, load_results, Geometry, RUNS_WIDE
from rd_theory import gchisq_cdf_imhof

cov, resd, out = sys.argv[1:4]
rng = np.random.default_rng(21)
KAPPA = 0.65 / 0.036
rep = {}


def man(run):
    return sorted(set(x['speaker'] for x in csv.DictReader(open(f'{cov}/{run}/step_2500/train_manifest.csv'))) - {'lj'})


def pcopy(X, cb, v0, T, tau, K=32, extra_D=0.0):
    """Gibbs; extra_D: rendering-error term added to the distance of each LibriTTS codeword."""
    V = np.vstack([v0[None], cb])
    lw = np.r_[np.log(K) + tau / (2 * T), np.full(len(cb), np.log(K))]
    d2 = ((X[:, None, :] - V[None]) ** 2).sum(-1)
    d2[:, 1:] += extra_D
    L = lw[None] - d2 / (2 * T)
    P = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    Y = P @ V
    D = X - v0
    rho = 1 - ((Y - v0) * D).sum(1) / (D * D).sum(1)
    return norm.cdf(KAPPA * (0.5 - rho))


def fit_tau(X, cb, v0, q, T, **kw):
    return brentq(lambda t: (1 - pcopy(X, cb, v0, T, t, **kw).mean()) - q, -400, 900)


# ---------- common space N=128 ----------
G = Geometry(load_latents(f'{cov}/{RUNS_WIDE[128]}/step_2500/latents.npz'))
idx = {s: i for i, s in enumerate(G.spk_ids)}
v0 = G.lj
spk, X = G.refs('external')
CB = {N: np.array([G.means[idx[s]] for s in man(r) if s in idx]) for N, r in RUNS_WIDE.items()}
RES = {N: load_results(f'{resd}/{r}/step_2500/strength_1.0/results.csv') for N, r in RUNS_WIDE.items()}
OBS = {N: np.array([RES[N][('external', s)]['copy'] for s in spk]) for N in RUNS_WIDE}
qobs = {N: float(1 - OBS[N].mean()) for N in OBS}

# ---- point 1a: T from the width of the per-voice transition (logistic in the slack, pooling N) ----
slack = {N: ((X - v0) ** 2).sum(1) - ((X[:, None] - CB[N][None]) ** 2).sum(-1).min(1) for N in CB}
s_all = np.concatenate([slack[N] for N in (8, 32, 128)])
y_all = np.concatenate([OBS[N] for N in (8, 32, 128)]) * 25


def nll(p):
    tau_, lw = p
    pr = np.clip(expit((s_all - tau_) / np.exp(lw)), 1e-9, 1 - 1e-9)
    return -(y_all * np.log(pr) + (25 - y_all) * np.log(1 - pr)).sum()
fit = minimize(nll, [115, np.log(10)], method='Nelder-Mead')
w = float(np.exp(fit.x[1]))
# bootstrap over voices
wb = []
for b in range(300):
    ii = rng.integers(0, 38, 38)
    s_b = np.concatenate([slack[N][ii] for N in (8, 32, 128)]); y_b = np.concatenate([OBS[N][ii] for N in (8, 32, 128)]) * 25
    f = minimize(lambda p: -(y_b * np.log(np.clip(expit((s_b - p[0]) / np.exp(p[1])), 1e-9, 1)) +
                             (25 - y_b) * np.log(np.clip(1 - expit((s_b - p[0]) / np.exp(p[1])), 1e-9, 1))).sum(),
                 fit.x, method='Nelder-Mead')
    wb.append(np.exp(f.x[1]))
rep['p1_width'] = dict(tau_logit=float(fit.x[0]), w=w, T_width=w / 2, w_ci=np.percentile(wb, [5, 95]).tolist(),
                      nota='w = 2T in the Gibbs channel (upper bound: includes geometry error)')

# ---- point 1b: test with T=1 fixed a priori; bootstrap of the residual (obs - pred) and of the null with no dependence on N ----
T0 = 1.0
tauE = fit_tau(X, CB[128], v0, qobs[128], T0)
pred = {N: float(1 - pcopy(X, CB[N], v0, T0, tauE).mean()) for N in CB}
res_b, null_b = [], []
for b in range(300):
    ii = rng.integers(0, 38, 38)
    q128b = 1 - OBS[128][ii].mean()
    try:
        tb = fit_tau(X[ii], CB[128], v0, q128b, T0)
    except ValueError:
        continue
    r = []
    for N in (8, 32):
        pb = 1 - pcopy(X[ii], CB[N], v0, T0, tb).mean()
        r.append((1 - OBS[N][ii].mean()) - pb)
    res_b.append(r)
    null_b.append([(1 - OBS[N][ii].mean()) - q128b for N in (8, 32)])
res_b, null_b = np.array(res_b), np.array(null_b)
rep['p1_test_T1'] = dict(tau=tauE, pred=pred, obs=qobs,
                         resid_ci90={8: np.percentile(res_b[:, 0], [5, 95]).tolist(), 32: np.percentile(res_b[:, 1], [5, 95]).tolist()},
                         null_noN_resid_ci90={8: np.percentile(null_b[:, 0], [5, 95]).tolist(), 32: np.percentile(null_b[:, 1], [5, 95]).tolist()},
                         frac_pred_explained={N: float((pred[N] - qobs[128]) / (qobs[N] - qobs[128])) for N in (8, 32)})

# ---- point 3: spaces. common space N=32-wide: fit at N=32, predict N=8; same in the 128 space ----
sp = {}
for base in (32, 128):
    Gb = Geometry(load_latents(f'{cov}/{RUNS_WIDE[base]}/step_2500/latents.npz'))
    ib = {s: i for i, s in enumerate(Gb.spk_ids)}
    sb, Xb = Gb.refs('external')
    assert list(sb) == list(spk)
    cbs = {N: np.array([Gb.means[ib[s]] for s in man(RUNS_WIDE[N]) if s in ib]) for N in (8, 32)}
    t32 = fit_tau(Xb, cbs[32], Gb.lj, qobs[32], T0)
    sp[f'espaco_{base}'] = dict(tau_fit_on32=t32, q8_pred=float(1 - pcopy(Xb, cbs[8], Gb.lj, T0, t32).mean()),
                                n_cb8=len(cbs[8]), n_cb32=len(cbs[32]),
                                spearman8=float(spearmanr(pcopy(Xb, cbs[8], Gb.lj, T0, t32), OBS[8])[0]))
# own space: each run in its own (a4: tau fitted per run). Here: tau of run 32 applied to run 8 in each own space
own = {}
for N, r in RUNS_WIDE.items():
    Go = Geometry(load_latents(f'{cov}/{r}/step_2500/latents.npz'))
    so, Xo = Go.refs('external')
    own[N] = dict(tau_fit=fit_tau(Xo, Go.means, Go.lj, qobs[N], T0), dLJc2=float(((Go.means.mean(0) - Go.lj) ** 2).sum()),
                  trSb=float(np.clip(np.linalg.eigvalsh(Go.Sb), 0, None).sum()))
sp['espaco_proprio_tau_por_run'] = own
rep['p3_espacos'] = sp

# ---- point 4: effective mass and M/2 ----
lnMef = np.log(32) + tauE / (2 * T0)
Tw = w / 2
tauW = fit_tau(X, CB[128], v0, qobs[128], Tw)
dq = {}
for lab, T_, ta in (('T1', T0, tauE), ('Twidth', Tw, tauW)):
    base_q = 1 - pcopy(X, CB[128], v0, T_, ta).mean()
    half_q = 1 - pcopy(X, CB[128], v0, T_, ta - 2 * T_ * np.log(2)).mean()
    bs = []
    for b in range(300):
        ii = rng.integers(0, 38, 38)
        try:
            tb = fit_tau(X[ii], CB[128], v0, 1 - OBS[128][ii].mean(), T_)
        except ValueError:
            continue
        bs.append((1 - pcopy(X[ii], CB[128], v0, T_, tb - 2 * T_ * np.log(2)).mean()) - (1 - OBS[128][ii].mean()))
    dq[lab] = dict(T=T_, dq=float(half_q - base_q), ci90=np.percentile(bs, [5, 95]).tolist(),
                   lnMef_over_K=float(ta / (2 * T_)), Mef=float(32 * np.exp(ta / (2 * T_))))
rep['p4_M'] = dict(lnMef=float(lnMef), Mef_T1=float(np.exp(lnMef)), meio_M=dq)

# ---- point 5a: the 4 fixed voices of the grid ----
su, Xu = G.refs('unseen')
f0u = {s: RES[128][('unseen', s)]['f0'] for s in su}
dlj2_u = ((Xu - v0) ** 2).sum(1)
g = {}
for i, s in enumerate(su):
    row = dict(f0=f0u[s], dLJ2=float(dlj2_u[i]), dLJ2_pct_among38=float(np.mean(((X - v0) ** 2).sum(1) < dlj2_u[i])),
               dCenter=float(np.linalg.norm(Xu[i] - G.center)))
    for N in (8, 32, 128):
        dnn = ((Xu[i] - CB[N]) ** 2).sum(1).min()
        sl = dlj2_u[i] - dnn
        row[f'dNN2_{N}'] = float(dnn); row[f'slack_{N}'] = float(sl)
        row[f'slack_pct_{N}'] = float(np.mean(slack[N] < sl))
        row[f'obs_copy_{N}'] = RES[N][('unseen', s)]['copy']
        row[f'pred_copy_{N}'] = float(pcopy(Xu[i:i + 1], CB[N], v0, T0, tauE)[0])
    g[s] = row
rep['p5a_vozes_grade'] = g
rep['p5a_38_ref'] = dict(f0_median=float(np.median([RES[8][('external', s)]['f0'] for s in spk])),
                         dLJ2_median=float(np.median(((X - v0) ** 2).sum(1))),
                         dCenter_median=float(np.median(np.linalg.norm(X - G.center, axis=1))),
                         slack_median={N: float(np.median(slack[N])) for N in slack})
rep['p5a_ref_ids'] = dict(unseen=[str(x) for x in G.lat['utt_id'][(G.lat['source'] == 'refs') & (G.lat['emotion'] == 'unseen')]],
                          external_ex=[str(x) for x in G.lat['utt_id'][(G.lat['source'] == 'refs') & (G.lat['emotion'] == 'external')][:5]])

# ---- point 5b: trained voices, q_seen(K) at N=8 ----
ss, Xs = G.refs('seen')
cb8 = CB[8]
own_idx = [man(RUNS_WIDE[8]).index(s) for s in ss]
obs_seen = {}
for K in (8, 32, 128):
    vals = []
    for sd in (1, 2):
        r = f'n8_k{K}_s{sd}' if not (K == 32 and sd == 2) else None
        if r is None:
            continue
        rr = load_results(f'{resd}/{r}/step_2500/strength_1.0/results.csv')
        vals.append(1 - np.mean([rr[('seen', s)]['copy'] for s in ss]))
    obs_seen[K] = vals
pred_seen = {}
tr_wx = float(np.trace(G.Wx))
for K in (8, 32, 128):
    qs = []
    for m in range(200):
        cbK = cb8 + rng.standard_normal(cb8.shape) * np.sqrt(max(1 / K - 1 / 32, 0) + 0.0)
        qs.append(1 - pcopy(Xs, cbK, v0, T0, tauE - 2 * T0 * np.log(K / 32), K=K).mean())
    pred_seen[K] = float(np.mean(qs))
# rendering term A/K added to the distance: fit A on the trained voices (K=8 vs 128), test on the new voices of the grid
def seen_q(A, K):
    return 1 - pcopy(Xs, cb8, v0, T0, tauE - 2 * T0 * np.log(K / 32) + A / 32, K=K, extra_D=A / K).mean()
# (tau re-anchored so that K=32 does not change: tau' = tau + A/32)
As = np.linspace(0, 3000, 61)
errs = [sum((seen_q(A, K) - np.mean(obs_seen[K])) ** 2 for K in (8, 32, 128)) for A in As]
Abest = float(As[int(np.argmin(errs))])
pred_seen_A = {K: float(seen_q(Abest, K)) for K in (8, 32, 128)}
# consequence for the externals and the new grid voices with K=8 and 128 (N=128)
ext_K = {K: float(1 - pcopy(X, CB[128], v0, T0, tauE - 2 * T0 * np.log(K / 32) + Abest / 32, K=K, extra_D=Abest / K).mean())
         for K in (8, 32, 128)}
un_K = {}
for r in ('n8_k8_s1', 'n8_k32_s1', 'n8_k128_s1', 'n32_k8_s1', 'n32_k32_s1', 'n32_k128_s1', 'n128_k8_s1', 'n128_k32_s1'):
    N = int(r.split('_')[0][1:]); K = int(r.split('_')[1][1:])
    cb = np.array([G.means[idx[s]] for s in man(r) if s in idx])
    rr = load_results(f'{resd}/{r}/step_2500/strength_1.0/results.csv')
    un_K[r] = dict(obs=float(1 - np.mean([rr[('unseen', s)]['copy'] for s in su])),
                   pred_A=float(1 - pcopy(Xu, cb, v0, T0, tauE - 2 * T0 * np.log(K / 32) + Abest / 32, K=K, extra_D=Abest / K).mean()),
                   pred_noA=float(1 - pcopy(Xu, cb, v0, T0, tauE - 2 * T0 * np.log(K / 32), K=K).mean()))
rep['p5b_seen'] = dict(obs=obs_seen, pred_codeword_noise=pred_seen, A_fit=Abest, pred_with_A=pred_seen_A,
                       ext38_N128_byK_with_A=ext_K, grade_unseen=un_K, trWx=tr_wx,
                       D_self_codeword_noise={K: tr_wx + 32 / K for K in (8, 32, 128)})

# ---- point 8: averaging of references on the 4 grid voices ----
c8 = {}
for i, s in enumerate(su):
    for N in (8, 128):
        j = ((Xu[i] - CB[N]) ** 2).sum(1).argmin()
        a = CB[N][j] - v0
        sd1 = float(2 * np.sqrt(a @ G.Wx @ a))
        c8[f'{s}_N{N}'] = dict(slack_minus_tau=float(slack_u := dlj2_u[i] - ((Xu[i] - CB[N][j]) ** 2).sum() - tauE),
                               sd_cross_R1=sd1, sd_cross_R4=sd1 / 2, z=float(slack_u / sd1))
sd38 = []
for i in range(38):
    j = ((X[i] - CB[128]) ** 2).sum(1).argmin(); a = CB[128][j] - v0
    sd38.append(2 * np.sqrt(a @ G.Wx @ a))
z38 = (slack[128] - tauE) / np.array(sd38)
rep['p8_R'] = dict(grade=c8, frac38_within_1sd=float(np.mean(np.abs(z38) < 1)), sd38_median=float(np.median(sd38)))

# ---- point 10: self-information L(c) per voice against distance ----
ev, U = np.linalg.eigh(G.Sb); ev = np.clip(ev, 0, None)
wv = ev + 1 / 32 + tr_wx / 32 / 8
t_c = ((X - v0) ** 2).sum(1) - tauE
nb = len(G.means)
d2b = ((X[:, None] - G.means[None]) ** 2).sum(-1)
kc = (d2b <= t_c[:, None]).sum(1)
pg = np.array([gchisq_cdf_imhof(t, wv, (U.T @ (c - G.center)) ** 2 / wv) if t > 0 else 0.0 for c, t in zip(X, t_c)])
p_emp = np.where(kc >= 3, kc / nb, pg)
Lg = -np.log2(np.maximum(pg, 1e-300)); Le = -np.log2(np.maximum(p_emp, 1e-300))
dLJ = np.sqrt(((X - v0) ** 2).sum(1))


def auc(score, y):
    pos, neg = score[y > 0.5], score[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(np.mean([(p > n_) + 0.5 * (p == n_) for p in pos for n_ in neg]))
p10 = {}
for N in (8, 32, 128):
    y = OBS[N]
    pr_L = 1 - np.exp(-N * np.maximum(pg, 0))
    pr_e = 1 - np.exp(-N * p_emp)
    p10[N] = dict(sp_minusLgauss=float(spearmanr(-Lg, y)[0]), sp_minusLemp=float(spearmanr(-Le, y)[0]),
                  sp_dLJ=float(spearmanr(dLJ, y)[0]), sp_slack=float(spearmanr(slack[N], y)[0]),
                  auc_Lgauss=auc(-Lg, y), auc_dLJ=auc(dLJ, y), auc_slack=auc(slack[N], y),
                  q_infospec_gauss=float(1 - pr_L.mean()),
                  brier_infospec=float(((pr_L - y) ** 2).mean()), brier_const=float(((y.mean() - y) ** 2).mean()))
rep['p10_L'] = p10
json.dump(rep, open(f'{out}/b1.json', 'w'), indent=1, default=float)
print(json.dumps(rep, indent=1, default=float))
