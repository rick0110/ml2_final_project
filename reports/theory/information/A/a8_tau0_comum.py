"""a8: prediction with tau0 = ||mu_128 - v0||^2 (no fitting) in the common N=128 space, T=1,2,3, real codebooks.
python3 -I a8_tau0_comum.py <coverage2> <results> <out>"""
import sys, os, json, csv
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
from rd_lib import load_latents, load_results, Geometry, RUNS_WIDE
cov, resd, out = sys.argv[1:4]
spec = importlib.util.spec_from_file_location('a7', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a7_grade.py'))
G = Geometry(load_latents(f'{cov}/{RUNS_WIDE[128]}/step_2500/latents.npz'))
idx = {s: i for i, s in enumerate(G.spk_ids)}
from scipy.stats import norm
from scipy.special import logsumexp
def pcopy(X, cb, T, tau, K=32):
    V = np.vstack([G.lj[None], cb]); lw = np.r_[np.log(K) + tau / (2 * T), np.full(len(cb), np.log(K))]
    L = lw[None] - ((X[:, None] - V[None]) ** 2).sum(-1) / (2 * T)
    P = np.exp(L - logsumexp(L, axis=1, keepdims=True)); Y = P @ V; D = X - G.lj
    rho = 1 - ((Y - G.lj) * D).sum(1) / (D * D).sum(1)
    return norm.cdf(0.65 / 0.036 * (0.5 - rho))
spk, X = G.refs('external')
tau0 = float(((G.center - G.lj) ** 2).sum())
rep = {'tau0': tau0}
for T in (1.0, 2.0, 3.0):
    r = {}
    for N, run in RUNS_WIDE.items():
        ss = sorted(set(x['speaker'] for x in csv.DictReader(open(f'{cov}/{run}/step_2500/train_manifest.csv'))) - {'lj'})
        cb = np.array([G.means[idx[s]] for s in ss if s in idx])
        p = pcopy(X, cb, T, tau0)
        obs = np.array([load_results(f'{resd}/{run}/step_2500/strength_1.0/results.csv')[('external', s)]['copy'] for s in spk])
        r[N] = [round(float(1 - p.mean()), 3), round(float(spearmanr(p, obs)[0]), 3)]
    rep[T] = r
print(json.dumps(rep)); json.dump(rep, open(f'{out}/a8.json', 'w'))
