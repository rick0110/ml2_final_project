"""a3: modelo 'livro-codigo + atrator' calibrado sem a curva de copia e teste de previsao.

Regra (limite de Poisson do canal de teste de Gibbs, ver relatorio sec. 3):
    nao copiada  <=>  D_N(c) = min_i ||c - v_i||^2  >  t(c) = ||c - v0||^2 - tau,
    tau = 2 T ln(M_ef / K).
Versao suave: canal de Gibbs com temperatura T e massa M_ef = K exp(tau/2T).

Calibracoes de tau (cada uma usa UMA quantidade):
  (S) vozes treinadas (8 'seen', presentes em todos os livros-codigo) -> preve as 38 externas em N=8,32,128
  (E128) externas em N=128 -> preve N=8, 32
Modelo populacional: c ~ N(mu, Sb) + ruido de referencia Wx; codewords ~ N(mu, Sb + I/K);
    q_pop(N) = E_c[(1 - F_c(t(c)))^N]   (F_c: qui-quadrado generalizado, Imhof)
python3 -I a3_modelo_codigo.py <coverage2> <results> <out>
"""
import sys, os, json, csv
import numpy as np
from scipy.stats import norm, spearmanr
from scipy.special import logsumexp
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_lib import load_latents, load_results, Geometry, RUNS_WIDE
from rd_theory import gchisq_cdf_imhof

cov, resd, out = sys.argv[1:4]
rng = np.random.default_rng(1)
G = Geometry(load_latents(f'{cov}/{RUNS_WIDE[128]}/step_2500/latents.npz'))
idx = {s: i for i, s in enumerate(G.spk_ids)}
v0 = G.lj
K = 32
KAPPA = 0.65 / 0.036


def codebook(run):
    rows = list(csv.DictReader(open(f'{cov}/{run}/step_2500/train_manifest.csv')))
    ss = sorted(set(x['speaker'] for x in rows) - {'lj'})
    have = [s for s in ss if s in idx]
    return np.array([G.means[idx[s]] for s in have]), len(ss), len(have), ss


def pcopy_gibbs(X, cb, T, tau, Kk=K):
    """canal de Gibbs; massa da LJ = K exp(tau/2T)."""
    V = np.vstack([v0[None], cb])
    lw = np.r_[np.log(Kk) + tau / (2 * T), np.full(len(cb), np.log(Kk))]
    d2 = ((X[:, None, :] - V[None]) ** 2).sum(-1)
    L = lw[None] - d2 / (2 * T)
    P = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    Y = P @ V
    D = X - v0
    rho = 1 - ((Y - v0) * D).sum(1) / (D * D).sum(1)
    return norm.cdf(KAPPA * (0.5 - rho))


def fit_tau(X, cb, qtarget, T):
    f = lambda tau: (1 - pcopy_gibbs(X, cb, T, tau).mean()) - qtarget
    return brentq(f, -200, 800)


ext_spk, Xe = G.refs('external')
seen_spk, Xs = G.refs('seen')
uns_spk, Xu = G.refs('unseen')
CB = {N: codebook(r)[0] for N, r in RUNS_WIDE.items()}
RES = {N: load_results(f'{resd}/{r}/step_2500/strength_1.0/results.csv') for N, r in RUNS_WIDE.items()}
qobs = {N: 1 - np.mean([RES[N][('external', s)]['copy'] for s in ext_spk]) for N in RUNS_WIDE}
qseen = {N: 1 - np.mean([RES[N][('seen', s)]['copy'] for s in seen_spk]) for N in RUNS_WIDE}
qseen_pool = float(np.mean(list(qseen.values())))
seen_in = [s in set(codebook(RUNS_WIDE[8])[3]) for s in seen_spk]
rep = dict(q_obs_ext=qobs, q_obs_seen=qseen, seen_in_N8_codebook=int(sum(seen_in)))

# estatistica de folga s(c) = ||c-v0||^2 - D_N(c)
def slack(X, cb):
    return ((X - v0) ** 2).sum(1) - ((X[:, None] - cb[None]) ** 2).sum(-1).min(1)
rep['slack_ext'] = {N: [float(np.percentile(slack(Xe, CB[N]), p)) for p in (10, 25, 50, 75, 90)] for N in CB}
rep['slack_seen'] = [float(x) for x in np.sort(slack(Xs, CB[8]))]
rep['dist2_LJ_ext_pct'] = [float(np.percentile(((Xe - v0) ** 2).sum(1), p)) for p in (5, 10, 25, 50, 75, 90)]

res_fit = {}
for T in (1.0, 2.0, 3.0):
    o = {}
    # (S) calibrado nas vozes treinadas (N=8 codebook contem as 8)
    tauS = fit_tau(Xs, CB[8], qseen_pool, T)
    o['tau_S'] = tauS
    o['pred_S'] = {N: float(1 - pcopy_gibbs(Xe, CB[N], T, tauS).mean()) for N in CB}
    o['pred_S_seen_byN'] = {N: float(1 - pcopy_gibbs(Xs, CB[N], T, tauS).mean()) for N in CB}
    # (E128)
    tauE = fit_tau(Xe, CB[128], qobs[128], T)
    o['tau_E128'] = tauE
    o['pred_E128'] = {N: float(1 - pcopy_gibbs(Xe, CB[N], T, tauE).mean()) for N in CB}
    o['pred_E128_seen'] = {N: float(1 - pcopy_gibbs(Xs, CB[N], T, tauE).mean()) for N in CB}
    o['spearman_E128'] = {N: float(spearmanr(pcopy_gibbs(Xe, CB[N], T, tauE),
                                                [RES[N][('external', s)]['copy'] for s in ext_spk])[0]) for N in CB}
    o['lnMef_E128'] = float(np.log(K) + tauE / (2 * T))
    res_fit[T] = o
rep['fits'] = res_fit

# ---------- modelo populacional (livro-codigo aleatorio gaussiano) ----------
ev, U = np.linalg.eigh(G.Sb); ev = np.clip(ev, 0, None)
mu = G.center
Wx = G.Wx


def q_pop_given_c(Xc, tau, Ns, Kk=K, cw_extra=0.0):
    """para cada c: P(D_N > t) = (1-F_c(t))^N, codewords ~ N(mu, Sb + (1/K + cw_extra) I)."""
    w = ev + 1.0 / Kk + cw_extra
    out_ = np.zeros((len(Xc), len(Ns)))
    for i, c in enumerate(Xc):
        a = U.T @ (c - mu)
        t = ((c - v0) ** 2).sum() - tau
        F = gchisq_cdf_imhof(t, w, a ** 2 / w) if t > 0 else 0.0
        out_[i] = np.exp(np.array(Ns) * np.log1p(-min(F, 1 - 1e-16)))
    return out_


Ns = [1, 2, 8, 32, 128, 247, 512, 1000, 2048, 2456, 10000, 100000]
tauE = res_fit[1.0]['tau_E128']
tauS = res_fit[1.0]['tau_S']
# (i) c = as 38 referencias reais, codebook gaussiano
for name, tau in (('E128', tauE), ('S', tauS)):
    Q = q_pop_given_c(Xe, tau, Ns, cw_extra=np.trace(Wx) / 32 / 8)
    rep[f'pop_realc_tau{name}'] = dict(zip(map(str, Ns), Q.mean(0).round(4).tolist()))
# (ii) c sorteado da populacao gaussiana
Sref = U @ np.diag(ev) @ U.T + Wx
Lc = np.linalg.cholesky(Sref + 1e-9 * np.eye(32))
Xc = mu + rng.standard_normal((1500, 32)) @ Lc.T
for name, tau in (('E128', tauE), ('S', tauS)):
    Q = q_pop_given_c(Xc, tau, Ns, cw_extra=np.trace(Wx) / 32 / 8)
    rep[f'pop_gaussc_tau{name}'] = dict(zip(map(str, Ns), Q.mean(0).round(4).tolist()))
    rep[f'pop_gaussc_tau{name}_floor'] = float(np.mean(((Xc - v0) ** 2).sum(1) < tau))
rep['floor_realc_tauE'] = float(np.mean(((Xe - v0) ** 2).sum(1) < tauE))
# variacoes: K, M
for lab, Kk, dtau in (('K8', 8, 2 * np.log(32 / 8)), ('K128', 128, -2 * np.log(4)), ('Mhalf', 32, -2 * np.log(2))):
    Q = q_pop_given_c(Xe, tauE + dtau, Ns, Kk=Kk, cw_extra=np.trace(Wx) / 32 / 8)
    rep[f'pop_realc_{lab}_T1'] = dict(zip(map(str, Ns), Q.mean(0).round(4).tolist()))
# D_N real vs gaussiano (teste da hipotese gaussiana para a distorcao)
dn = {}
for N in CB:
    real = ((Xe[:, None] - CB[N][None]) ** 2).sum(-1).min(1)
    sims = []
    for r in range(200):
        cbg = mu + rng.standard_normal((N, 32)) @ (U @ np.diag(np.sqrt(ev + 1 / K + np.trace(Wx) / 32 / 8))).T
        sims.append(np.median(((Xe[:, None] - cbg[None]) ** 2).sum(-1).min(1)))
    dn[N] = dict(real_median=float(np.median(real)), gauss_median=float(np.mean(sims)),
                 gauss_sd=float(np.std(sims)))
rep['DN_real_vs_gauss'] = dn
json.dump(rep, open(f'{out}/a3.json', 'w'), indent=1, default=float)
print(json.dumps(rep, indent=1, default=float))
