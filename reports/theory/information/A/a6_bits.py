"""a6: bits por voz. Para cada referencia externa: tolerancia t(c) = ||c-v0||^2 - tau (tau da regra dura, a5),
L(c) = -log2 P_v(||c-V||^2 <= t) (livro-codigo aleatorio, cauda gaussiana; = log2 do N em que a voz passa a ser
copiada com prob. 1-1/e), e R_wf(t) = taxa de agua reversa da fonte das referencias (espectro de Sb+Wx) em D=t
(cota inferior: nenhum livro-codigo de 2^R vozes faz melhor em media). Tambem a decomposicao em direcoes.
python3 -I a6_bits.py <coverage2> <out>
"""
import sys, os, json
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_lib import load_latents, Geometry, RUNS_WIDE
from rd_theory import gchisq_cdf_imhof, waterfill

cov, out = sys.argv[1:3]
G = Geometry(load_latents(f'{cov}/{RUNS_WIDE[128]}/step_2500/latents.npz'))
a5 = json.load(open(f'{out}/a5.json'))
tau = a5['tau_hard']
spk, X = G.refs('external')
ev, U = np.linalg.eigh(G.Sb); ev = np.clip(ev, 0, None)
mu, v0 = G.center, G.lj
w = ev + 1 / 32 + np.trace(G.Wx) / 32 / 8
lam_c = np.clip(np.linalg.eigvalsh(G.Sb + G.Wx), 1e-9, None)
rows = []
for c in X:
    t = ((c - v0) ** 2).sum() - tau
    a = U.T @ (c - mu)
    p = gchisq_cdf_imhof(t, w, a ** 2 / w) if t > 0 else 0.0
    L = -np.log2(p) if p > 0 else np.inf
    if t <= 0:
        R = np.inf
    elif t >= lam_c.sum():
        R = 0.0
    else:
        R = brentq(lambda r: waterfill(lam_c, r)[0] - t, 0, 400)
    rows.append((t, L, R))
rows = np.array(rows)
fin = np.isfinite(rows[:, 1])
rep = dict(tau=tau,
           L_q=[float(np.quantile(np.where(fin, rows[:, 1], 1e9), q)) for q in (0.25, 0.5, 0.75, 0.8, 0.85, 0.9)],
           R_q=[float(np.quantile(np.where(np.isfinite(rows[:, 2]), rows[:, 2], 1e9), q)) for q in (0.25, 0.5, 0.75, 0.8, 0.85, 0.9)],
           penalty_median_bits=float(np.median((rows[:, 1] - rows[:, 2])[fin & (rows[:, 2] > 0)])),
           n_inf=int((~fin).sum()),
           cum_bits_voice_dirs=np.cumsum(0.5 * np.log2(1 + np.sort(ev)[::-1]))[:12].round(2).tolist())
json.dump(rep, open(f'{out}/a6.json', 'w'), indent=1)
print(json.dumps(rep, indent=1))
