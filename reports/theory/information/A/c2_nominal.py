"""c2 (v3): predictions with the NOMINAL level lam = ln(M/K) = ln(2000/32) (no level parameter fitted)
and shape T: (i) the run's own T, (ii) common T, (iii) T from the other runs; decomposition with nominal lam.
python3 -I c2_nominal.py <coverage2> <results> <out>"""
import sys, os, json, csv
import numpy as np
from scipy.special import expit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
from rd_lib import load_latents, load_results, Geometry, RUNS_WIDE
cov, resd, out = sys.argv[1:4]
c1 = json.load(open(f'{out}/c1.json'))
G = Geometry(load_latents(f'{cov}/{RUNS_WIDE[128]}/step_2500/latents.npz'))
idx = {s: i for i, s in enumerate(G.spk_ids)}
v0 = G.lj
from scipy.special import logsumexp
man = lambda run: sorted(set(x['speaker'] for x in csv.DictReader(open(f'{cov}/{run}/step_2500/train_manifest.csv'))) - {'lj'})
spk, X = G.refs('external')
CB = {N: np.array([G.means[idx[s]] for s in man(r) if s in idx]) for N, r in RUNS_WIDE.items()}
OBS = {N: np.array([load_results(f'{resd}/{r}/step_2500/strength_1.0/results.csv')[('external', s)]['copy'] for s in spk]) for N, r in RUNS_WIDE.items()}
def q(Xc, cb, T, lam):
    d0 = ((Xc - v0) ** 2).sum(1); di = ((Xc[:, None] - cb[None]) ** 2).sum(-1)
    return float(1 - expit(logsumexp(-di / (2 * T), axis=1) + d0 / (2 * T) - lam).mean())
lamN = float(np.log(2000 / 32))
T = {N: c1['wide'][str(N)]['T'] for N in (8, 32, 128)}
Tc = c1['T_comum']['T']
rep = dict(lam_nominal=lamN, T=T, T_comum=Tc)
for N in (8, 32, 128):
    To = float(np.exp(np.mean([np.log(T[M]) for M in (8, 32, 128) if M != N])))
    bts = np.array(c1['wide'][str(N)]['boot_T'])
    rep[N] = dict(q_obs=float(1 - OBS[N].mean()), q_Tproprio=q(X, CB[N], T[N], lamN), q_Tcomum=q(X, CB[N], Tc, lamN),
                  q_Toutros=q(X, CB[N], To, lamN), T_outros=To,
                  q_Tproprio_ci90=[q(X, CB[N], t, lamN) for t in np.percentile(bts, [5, 95])],
                  q_Tcomum_K8=q(X, CB[N], Tc, float(np.log(2000 / 8))), q_Tcomum_K128=q(X, CB[N], Tc, float(np.log(2000 / 128))),
                  q_Tcomum_Mhalf=q(X, CB[N], Tc, float(np.log(1000 / 32))))
# decomposition with nominal lam and the run's own T
Q = lambda a, b: q(X, CB[a], T[b], lamN)
dec = {}
for a, b in ((8, 32), (32, 128), (8, 128)):
    tot = Q(b, b) - Q(a, a)
    cap = 0.5 * ((Q(a, b) - Q(a, a)) + (Q(b, b) - Q(b, a)))
    dec[f'{a}->{b}'] = dict(total=tot, cobertura=tot - cap, capacidade=cap)
rep['dec_nominal'] = dec
# Tc: sweep T for the hypothetical capacity effect: q(cb_N, T) for T = 5, 10, 20, 40
rep['varre_T'] = {t: {N: q(X, CB[N], t, lamN) for N in (8, 32, 128)} for t in (2, 5, 10, 20, 40)}
# capacity C(T) and the T needed for q=0.1 with cb_128
ev = np.clip(np.linalg.eigvalsh(G.Sb), 0, None)
rep['C_of_T'] = {t: float(0.5 * np.sum(np.log2(1 + ev / t))) for t in (1, 2, 5, 10, 20, 27)}
# extrapolation in N: sampled Gaussian codebook (measured spectrum), common T, nominal lam
evv, U = np.linalg.eigh(G.Sb); evv = np.clip(evv, 0, None)
rng = np.random.default_rng(9)
ext = {}
for N in (128, 512, 2048, 2456):
    for t in (Tc, 10.0, 5.0):
        vals = []
        for r in range(6 if N > 600 else 20):
            cb = G.center + (rng.standard_normal((N, 32)) * np.sqrt(evv + 1 / 32)) @ U.T
            vals.append(q(X, cb, t, lamN))
        ext[f'N{N}_T{t:.1f}'] = [float(np.mean(vals)), float(np.std(vals))]
rep['extrap_gauss'] = ext
print(json.dumps(rep, indent=1)); json.dump(rep, open(f'{out}/c2.json', 'w'), indent=1)
