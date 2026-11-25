"""c3 (v3): extrapolacao com T comum (forma) e nivel lam(N) = a + b ln N ajustado nos 3 treinos -wide (T comum);
livro-codigo gaussiano (espectro medido) e livro-codigo empirico (127 vozes; so N<=127).
python3 -I c3_extrap.py <coverage2> <out>"""
import sys, os, json
import numpy as np
from scipy.special import expit, logsumexp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_lib import load_latents, Geometry, RUNS_WIDE
cov, out = sys.argv[1:3]
c1 = json.load(open(f'{out}/c1.json'))
G = Geometry(load_latents(f'{cov}/{RUNS_WIDE[128]}/step_2500/latents.npz'))
spk, X = G.refs('external'); v0 = G.lj
Tc = c1['T_comum']['T']; lams = c1['T_comum']['lams']
b, a = np.polyfit(np.log([8, 32, 128]), lams, 1)
def q(cb, T, lam):
    d0 = ((X - v0) ** 2).sum(1); di = ((X[:, None] - cb[None]) ** 2).sum(-1)
    return float(1 - expit(logsumexp(-di / (2 * T), axis=1) + d0 / (2 * T) - lam).mean())
ev, U = np.linalg.eigh(G.Sb); ev = np.clip(ev, 0, None)
rng = np.random.default_rng(3)
rep = dict(T=Tc, lam_a=float(a), lam_b=float(b), gamma_mass=float(1 - b), res={})
for N in (8, 32, 128, 512, 2048, 2456, 10000):
    lam = a + b * np.log(N)
    g = [q(G.center + (rng.standard_normal((N, 32)) * np.sqrt(ev + 1 / 32)) @ U.T, Tc, lam) for _ in range(8)]
    e = [q(G.means[rng.choice(len(G.means), N, replace=False)], Tc, lam) for _ in range(30)] if N <= 127 else None
    rep['res'][N] = dict(lam=float(lam), q_gauss=float(np.mean(g)), q_emp=(float(np.mean(e)) if e else None),
                         q_gauss_Tc_nominal_lam=None)
print(json.dumps(rep, indent=1)); json.dump(rep, open(f'{out}/c3.json', 'w'), indent=1)
