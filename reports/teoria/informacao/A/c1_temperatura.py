"""c1 (v3): temperatura por treino T(N) pelo FORMATO da resposta por voz no canal de Gibbs.
Para cada treino r: P(copia | c) = 1 - pi_0(c),  pi_0 = M e^{-d0^2/2T} / (M e^{-d0^2/2T} + K sum_i e^{-di^2/2T}),
parametros (T_r, lam_r = ln(M_ef/K)); verossimilhanca binomial (25 frases por voz). T_r = forma, lam_r = nivel.
Capacidade usada: C(T) = 1/2 sum_j log2(1 + lam_j / T)  (Gibbs a temperatura T = Bayes com ruido T por direcao).
Previsao deixando o treino de fora: T_r proprio (forma) + nivel (lam ou tau=2T lam) dos OUTROS treinos.
Decomposicao cobertura x capacidade: q(cb_N, T_N') contrafactual.
python3 -I c1_temperatura.py <coverage2> <results> <out>
"""
import sys, os, json, csv
import numpy as np
from scipy.special import logsumexp, expit
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_lib import load_latents, load_results, Geometry, RUNS_WIDE

cov, resd, out = sys.argv[1:4]
rng = np.random.default_rng(44)
G = Geometry(load_latents(f'{cov}/{RUNS_WIDE[128]}/step_2500/latents.npz'))
idx = {s: i for i, s in enumerate(G.spk_ids)}
v0 = G.lj
ev = np.clip(np.linalg.eigvalsh(G.Sb), 0, None)
man = lambda run: sorted(set(x['speaker'] for x in csv.DictReader(open(f'{cov}/{run}/step_2500/train_manifest.csv'))) - {'lj'})
C_bits = lambda T: float(0.5 * np.sum(np.log2(1 + ev / T)))


def logit_copy(X, cb, T, lam, K):
    """log-odds de copia = ln(K sum e^{-di^2/2T}) - ln(M e^{-d0^2/2T}),  lam = ln(M/K)."""
    d0 = ((X - v0) ** 2).sum(1)
    di = ((X[:, None] - cb[None]) ** 2).sum(-1)
    return logsumexp(-di / (2 * T), axis=1) + d0 / (2 * T) - lam


def nll(params, X, cb, y, K, n=25):
    T, lam = np.exp(params[0]), params[1]
    p = np.clip(expit(logit_copy(X, cb, T, lam, K)), 1e-6, 1 - 1e-6)
    return -(y * n * np.log(p) + (1 - y) * n * np.log(1 - p)).sum()


def fit(X, cb, y, K, x0=None):
    best = None
    starts = [(np.log(x0[0]), x0[1])] if x0 is not None else [(lT, lam) for lT in np.log([1, 3, 10, 30, 100]) for lam in (0, 5, 20, 60)]
    for lT, lam in starts:
        if True:
            r = minimize(nll, [lT, lam], args=(X, cb, y, K), method='Nelder-Mead',
                         options=dict(xatol=1e-4, fatol=1e-6, maxiter=3000))
            if best is None or r.fun < best.fun:
                best = r
    return float(np.exp(best.x[0])), float(best.x[1]), float(best.fun)


def qpred(X, cb, T, lam, K):
    return float(1 - expit(logit_copy(X, cb, T, lam, K)).mean())


# ---------- treinos -wide (38 externas) ----------
spk, X = G.refs('external')
CB = {N: np.array([G.means[idx[s]] for s in man(r) if s in idx]) for N, r in RUNS_WIDE.items()}
OBS = {N: np.array([load_results(f'{resd}/{r}/step_2500/strength_1.0/results.csv')[('external', s)]['copy'] for s in spk])
       for N, r in RUNS_WIDE.items()}
rep = {'wide': {}}
fits = {}
for N in (8, 32, 128):
    T, lam, f = fit(X, CB[N], OBS[N], 32)
    bs = []
    for b in range(150):
        ii = rng.integers(0, 38, 38)
        Tb, lb, _ = fit(X[ii], CB[N], OBS[N][ii], 32, x0=(T, lam))
        bs.append((Tb, lb))
    bs = np.array(bs)
    fits[N] = (T, lam)
    rep['wide'][N] = dict(T=T, T_ci90=np.percentile(bs[:, 0], [5, 95]).tolist(), T_median_boot=float(np.median(bs[:, 0])),
                          lam=lam, tau=2 * T * lam, C_bits=C_bits(T), C_ci90=[C_bits(x) for x in np.percentile(bs[:, 0], [95, 5])],
                          frac_evidence=1 / T, q_obs=float(1 - OBS[N].mean()), q_fit=qpred(X, CB[N], T, lam, 32),
                          boot_T=bs[:, 0].tolist())
    print(N, {k: v for k, v in rep['wide'][N].items() if k != 'boot_T'}, flush=True)
# tendencia: P(T8 > T128) no bootstrap
b8, b128 = np.array(rep['wide'][8]['boot_T']), np.array(rep['wide'][128]['boot_T'])
rep['P_T8_gt_T128'] = float(np.mean(b8[:, None] > b128[None, :]))
# ajuste conjunto com T comum (para comparar verossimilhanca)
def nll_common(p, shareT=True):
    tot = 0
    for j, N in enumerate((8, 32, 128)):
        lT = p[0] if shareT else p[j]
        lam = p[3] if shareT else p[3 + j] if False else p[1 + j] if shareT else p[3 + j]
        tot += nll([lT, lam], X, CB[N], OBS[N], 32)
    return tot
best = None
for lT in np.log([3, 10, 30]):
    for lam in (5, 20):
        r = minimize(lambda p: sum(nll([p[0], p[1 + j]], X, CB[N], OBS[N], 32) for j, N in enumerate((8, 32, 128))),
                     [lT, lam, lam, lam], method='Nelder-Mead', options=dict(maxiter=8000, xatol=1e-4, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
nll_sep = sum(nll([np.log(fits[N][0]), fits[N][1]], X, CB[N], OBS[N], 32) for N in (8, 32, 128))
rep['T_comum'] = dict(T=float(np.exp(best.x[0])), lams=best.x[1:].tolist(), nll_comum=float(best.fun), nll_sep=float(nll_sep),
                      LR=float(2 * (best.fun - nll_sep)), nota='LR ~ chi2(2) sob T comum, mas binomial superdispersa (x15) => dividir por ~15')
# ---------- previsao deixando o treino de fora ----------
loro = {}
for N in (8, 32, 128):
    others = [M for M in (8, 32, 128) if M != N]
    T = fits[N][0]
    lam_o = float(np.mean([fits[M][1] for M in others]))
    tau_o = float(np.mean([2 * fits[M][0] * fits[M][1] for M in others]))
    loro[N] = dict(T_proprio=T, q_lam_outros=qpred(X, CB[N], T, lam_o, 32), q_tau_outros=qpred(X, CB[N], T, tau_o / (2 * T), 32),
                   q_obs=float(1 - OBS[N].mean()))
    # T previsto pela tendencia log T ~ log N dos outros dois (sem nada do treino N)
    lt = np.polyfit(np.log(others), np.log([fits[M][0] for M in others]), 1)
    Tx = float(np.exp(np.polyval(lt, np.log(N))))
    loro[N]['T_extrap_outros'] = Tx
    loro[N]['q_tudo_dos_outros_lam'] = qpred(X, CB[N], Tx, lam_o, 32)
    loro[N]['q_tudo_dos_outros_tau'] = qpred(X, CB[N], Tx, tau_o / (2 * Tx), 32)
rep['loro'] = loro
print('loro', loro, flush=True)
# ---------- decomposicao cobertura x capacidade (nivel lam fixo = media; e tau fixo) ----------
dec = {}
for mode in ('lam', 'tau'):
    lam_bar = float(np.mean([fits[N][1] for N in fits])); tau_bar = float(np.mean([2 * fits[N][0] * fits[N][1] for N in fits]))
    def Q(Ncb, NT):
        T = fits[NT][0]
        lam = lam_bar if mode == 'lam' else tau_bar / (2 * T)
        return qpred(X, CB[Ncb], T, lam, 32)
    d = {}
    for a, b in ((8, 128), (8, 32), (32, 128)):
        tot = Q(b, b) - Q(a, a)
        cov_ = 0.5 * ((Q(b, a) - Q(a, a)) + (Q(b, b) - Q(a, b)))
        cap = 0.5 * ((Q(a, b) - Q(a, a)) + (Q(b, b) - Q(b, a)))
        d[f'{a}->{b}'] = dict(total=tot, cobertura=cov_, capacidade=cap, q=[Q(a, a), Q(b, a), Q(a, b), Q(b, b)])
    dec[mode] = d
# bootstrap da decomposicao 8->128 (modo lam), refazendo os ajustes
bd = []
for b in range(60):
    ii = rng.integers(0, 38, 38)
    fb = {N: fit(X[ii], CB[N], OBS[N][ii], 32, x0=fits[N])[:2] for N in (8, 128)}
    lb = 0.5 * (fb[8][1] + fb[128][1])
    Qb = lambda Ncb, NT: qpred(X[ii], CB[Ncb], fb[NT][0], lb, 32)
    tot = Qb(128, 128) - Qb(8, 8)
    cap = 0.5 * ((Qb(8, 128) - Qb(8, 8)) + (Qb(128, 128) - Qb(128, 8)))
    bd.append((tot, cap, tot - cap))
bd = np.array(bd)
dec['boot_8_128_lam'] = dict(total_ci=np.percentile(bd[:, 0], [5, 95]).tolist(), cap_ci=np.percentile(bd[:, 1], [5, 95]).tolist(),
                             cov_ci=np.percentile(bd[:, 2], [5, 95]).tolist(),
                             frac_cap_median=float(np.median(bd[:, 1] / np.where(np.abs(bd[:, 0]) > 1e-3, bd[:, 0], np.nan))))
rep['decomposicao'] = dec
print('dec', json.dumps(dec)[:2000], flush=True)
# ---------- grade (4 novas + 8 treinadas por treino) ----------
su, Xu = G.refs('unseen'); ss, Xs = G.refs('seen')
Xg = np.vstack([Xu, Xs])
grid = {}
for r in ('n8_k8_s1', 'n8_k8_s2', 'n8_k32_s1', 'n8_k128_s1', 'n8_k128_s2', 'n32_k8_s1', 'n32_k8_s2', 'n32_k32_s1',
          'n32_k128_s1', 'n32_k128_s2', 'n128_k8_s1', 'n128_k32_s1', 'n128_k32_s2'):
    N = int(r.split('_')[0][1:]); K = int(r.split('_')[1][1:])
    cb = np.array([G.means[idx[s]] for s in man(r) if s in idx])
    rr = load_results(f'{resd}/{r}/step_2500/strength_1.0/results.csv')
    y = np.array([rr[('unseen', s)]['copy'] for s in su] + [rr[('seen', s)]['copy'] for s in ss])
    T, lam, _ = fit(Xg, cb, y, K)
    bs = [fit(Xg[ii], cb, y[ii], K, x0=(T, lam))[0] for ii in (rng.integers(0, 12, 12) for _ in range(40))]
    grid[r] = dict(N=N, K=K, T=T, T_ci90=np.percentile(bs, [5, 95]).tolist(), lam=lam, C_bits=C_bits(T))
    print(r, grid[r], flush=True)
rep['grade'] = grid
json.dump(rep, open(f'{out}/c1.json', 'w'), indent=1, default=float)
