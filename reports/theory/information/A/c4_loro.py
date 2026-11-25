"""c4 (v3): leave-one-run-out prediction, with shape T from the run itself (per-voice width) or a common T,
and level lam extrapolated/interpolated in ln N from the OTHER two runs (per-run fits from c1).
python3 -I c4_loro.py <coverage2> <results> <out>"""
import sys, os, json, csv
import numpy as np
from scipy.special import expit, logsumexp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_lib import load_latents, load_results, Geometry, RUNS_WIDE
cov, resd, out = sys.argv[1:4]
c1 = json.load(open(f'{out}/c1.json'))
G = Geometry(load_latents(f'{cov}/{RUNS_WIDE[128]}/step_2500/latents.npz'))
idx = {s: i for i, s in enumerate(G.spk_ids)}; v0 = G.lj
man = lambda run: sorted(set(x['speaker'] for x in csv.DictReader(open(f'{cov}/{run}/step_2500/train_manifest.csv'))) - {'lj'})
spk, X = G.refs('external')
CB = {N: np.array([G.means[idx[s]] for s in man(r) if s in idx]) for N, r in RUNS_WIDE.items()}
OBS = {N: 1 - np.mean([load_results(f'{resd}/{r}/step_2500/strength_1.0/results.csv')[('external', s)]['copy'] for s in spk]) for N, r in RUNS_WIDE.items()}
def q(cb, T, lam):
    d0 = ((X - v0) ** 2).sum(1); di = ((X[:, None] - cb[None]) ** 2).sum(-1)
    return float(1 - expit(logsumexp(-di / (2 * T), axis=1) + d0 / (2 * T) - lam).mean())
Ns = (8, 32, 128)
lam_own = {N: c1['wide'][str(N)]['lam'] for N in Ns}; T_own = {N: c1['wide'][str(N)]['T'] for N in Ns}
Tc = c1['T_comum']['T']; lam_c = dict(zip(Ns, c1['T_comum']['lams']))
rep = {}
for N in Ns:
    o = [M for M in Ns if M != N]
    def trend(lams):
        b, a = np.polyfit(np.log(o), [lams[M] for M in o], 1)
        return a + b * np.log(N)
    rep[N] = dict(obs=OBS[N], q_Tproprio_lamtrend=q(CB[N], T_own[N], trend(lam_own)),
                  q_Tcomum_lamtrend=q(CB[N], Tc, trend(lam_c)), lam_trend_c=float(trend(lam_c)), lam_fit_c=lam_c[N])
print(json.dumps(rep, indent=1)); json.dump(rep, open(f'{out}/c4.json', 'w'), indent=1)
