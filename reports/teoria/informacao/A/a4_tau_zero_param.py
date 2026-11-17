"""a4: teste do limiar sem parametro livre tau = ||mu_N - v0||^2 em CADA run, no espaco do proprio run.
Para cada run: branqueia pelo W do proprio run, livro-codigo = medias dos locutores treinados presentes nos
latentes, v0 = media LJ, mu_N = media das medias treinadas. Preve q para as referencias 'unseen' (4),
'external' (38 nos -wide, 4 nos demais) e 'seen' (8); compara com o observado. Tambem ajusta tau por run.
python3 -I a4_tau_zero_param.py <coverage2> <results> <out>
"""
import sys, os, json, glob
import numpy as np
from scipy.stats import norm, spearmanr
from scipy.special import logsumexp
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_lib import load_latents, load_results, Geometry

cov, resd, out = sys.argv[1:4]
KAPPA = 0.65 / 0.036
runs = ['n8_k8_s1', 'n8_k8_s2', 'n8_k32_s1', 'n8_k128_s1', 'n8_k128_s2', 'n32_k8_s1', 'n32_k8_s2', 'n32_k32_s1',
        'n32_k128_s1', 'n32_k128_s2', 'n128_k8_s1', 'n128_k32_s1', 'n128_k32_s2', 'n200_k32_s1',
        'n8_k32_s1-wide', 'n32_k32_s1-wide', 'n128_k32_s1-wide']


def pcopy(X, cb, v0, T, tau, K):
    V = np.vstack([v0[None], cb])
    lw = np.r_[np.log(K) + tau / (2 * T), np.full(len(cb), np.log(K))]
    d2 = ((X[:, None, :] - V[None]) ** 2).sum(-1)
    L = lw[None] - d2 / (2 * T)
    P = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    Y = P @ V
    D = X - v0
    rho = 1 - ((Y - v0) * D).sum(1) / (D * D).sum(1)
    return norm.cdf(KAPPA * (0.5 - rho))


rows = []
for r in runs:
    N = int(r.split('_')[0][1:]); K = int(r.split('_')[1][1:])
    G = Geometry(load_latents(f'{cov}/{r}/step_2500/latents.npz'))
    res = load_results(f'{resd}/{r}/step_2500/strength_1.0/results.csv')
    cb = G.means
    mu = cb.mean(0)
    tau0 = float(((mu - G.lj) ** 2).sum())
    row = dict(run=r, N=N, K=K, n_cb=len(cb), tau0=tau0)
    for kind in ('unseen', 'external', 'seen'):
        spk, X = G.refs(kind)
        if len(spk) == 0:
            continue
        obs = np.array([res[(kind, s)]['copy'] for s in spk])
        row[f'{kind}_n'] = len(spk)
        row[f'{kind}_qobs'] = float(1 - obs.mean())
        for T in (1.0, 2.0, 3.0):
            p = pcopy(X, cb, G.lj, T, tau0, K)
            row[f'{kind}_qpred_T{int(T)}'] = float(1 - p.mean())
        try:
            row[f'{kind}_taufit_T2'] = brentq(lambda t: (1 - pcopy(X, cb, G.lj, 2.0, t, K).mean()) - (1 - obs.mean()),
                                              -300, 900)
        except ValueError:
            row[f'{kind}_taufit_T2'] = np.nan
        if kind == 'external' and len(spk) > 10:
            p = pcopy(X, cb, G.lj, 2.0, tau0, K)
            row['ext_spearman_T2'] = float(spearmanr(p, obs)[0])
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(f'{out}/a4.csv', index=False)
# resumo: unseen nos 14 runs base (4 vozes), RMSE e correlacao
base = df[~df.run.str.contains('wide')]
for T in (1, 2, 3):
    e = base[f'unseen_qpred_T{T}'] - base['unseen_qobs']
    print(f'unseen base T={T}: RMSE {np.sqrt((e**2).mean()):.3f} bias {e.mean():+.3f} corr {np.corrcoef(base[f"unseen_qpred_T{T}"], base.unseen_qobs)[0,1]:.3f}')
    e = base[f'seen_qpred_T{T}'] - base['seen_qobs']
    print(f'seen   base T={T}: RMSE {np.sqrt((e**2).mean()):.3f} bias {e.mean():+.3f} corr {np.corrcoef(base[f"seen_qpred_T{T}"], base.seen_qobs)[0,1]:.3f}')
