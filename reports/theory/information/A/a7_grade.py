"""a7: grade N x K (4 vozes novas, 8 treinadas) prevista no espaco comum do run N=128 com tau calibrado
nas 38 externas em N=128 (a3) e a dependencia em K pela teoria: tau(K) = tau32 - 2T ln(K/32),
ruido de codeword 1/K. Sem nenhum ajuste a grade.
python3 -I a7_grade.py <coverage2> <results> <out>
"""
import sys, os, json, csv
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_lib import load_latents, load_results, Geometry, RUNS_WIDE

cov, resd, out = sys.argv[1:4]
G = Geometry(load_latents(f'{cov}/{RUNS_WIDE[128]}/step_2500/latents.npz'))
idx = {s: i for i, s in enumerate(G.spk_ids)}
v0 = G.lj
KAPPA = 0.65 / 0.036
a3 = json.load(open(f'{out}/a3.json'))
rng = np.random.default_rng(5)


def pcopy(X, cb, T, tau, K):
    V = np.vstack([v0[None], cb])
    lw = np.r_[np.log(K) + tau / (2 * T), np.full(len(cb), np.log(K))]
    d2 = ((X[:, None, :] - V[None]) ** 2).sum(-1)
    L = lw[None] - d2 / (2 * T)
    P = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    Y = P @ V
    D = X - v0
    rho = 1 - ((Y - v0) * D).sum(1) / (D * D).sum(1)
    return norm.cdf(KAPPA * (0.5 - rho))


runs = ['n8_k8_s1', 'n8_k8_s2', 'n8_k32_s1', 'n8_k128_s1', 'n8_k128_s2', 'n32_k8_s1', 'n32_k8_s2', 'n32_k32_s1',
        'n32_k128_s1', 'n32_k128_s2', 'n128_k8_s1', 'n128_k32_s1', 'n128_k32_s2', 'n200_k32_s1']
rows = []
for r in runs:
    N = int(r.split('_')[0][1:]); K = int(r.split('_')[1][1:])
    man = list(csv.DictReader(open(f'{cov}/{r}/step_2500/train_manifest.csv')))
    ss = sorted(set(x['speaker'] for x in man) - {'lj'})
    have = [s for s in ss if s in idx]
    cb = np.array([G.means[idx[s]] for s in have])
    # locutores ausentes do espaco comum (n200): completar com sorteio gaussiano da populacao
    miss = len(ss) - len(have)
    if miss > 0:
        ev, U = np.linalg.eigh(G.Sb); ev = np.clip(ev, 0, None)
        cb = np.vstack([cb, G.center + rng.standard_normal((miss, 32)) @ (U @ np.diag(np.sqrt(ev))).T])
    cb = cb + rng.standard_normal(cb.shape) * np.sqrt(max(1 / K - 1 / 32, 0))
    res = load_results(f'{resd}/{r}/step_2500/strength_1.0/results.csv')
    row = dict(run=r, N=N, K=K, missing=miss)
    for kind in ('unseen', 'seen'):
        spk, X = G.refs(kind)
        obs = np.array([res[(kind, s)]['copy'] for s in spk])
        row[f'{kind}_obs'] = float(1 - obs.mean())
        for T in (1, 3):
            tau = a3['fits'][f'{float(T)}']['tau_E128'] - 2 * T * np.log(K / 32)
            row[f'{kind}_pred_T{T}'] = float(1 - pcopy(X, cb, T, tau, K).mean())
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
import pandas as pd
df = pd.DataFrame(rows); df.to_csv(f'{out}/a7.csv', index=False)
for kind in ('unseen', 'seen'):
    for T in (1, 3):
        e = df[f'{kind}_pred_T{T}'] - df[f'{kind}_obs']
        print(kind, T, 'RMSE', round(float(np.sqrt((e ** 2).mean())), 3), 'bias', round(float(e.mean()), 3),
              'corr', round(float(np.corrcoef(df[f'{kind}_pred_T{T}'], df[f'{kind}_obs'])[0, 1]), 3))
