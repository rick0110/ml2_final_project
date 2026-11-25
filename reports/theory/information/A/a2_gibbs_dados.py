"""a2: Gibbs decoder (Blahut-Arimoto test channel restricted to the training codebook)
applied to the 38 external references with the REAL nested codebooks (N=8 c 32 c 128),
all in the whitened space of the N=128 run.

  pi_s(c) ∝ n_s exp(-||c - v_s||^2 / 2T),  n_s = K (LibriTTS) or M_ef (LJ)
  y(c)    = sum_s pi_s v_s
  rho(c)  = 1 - <y - v0, c - v0>/||c - v0||^2        (Lemma 1 in the latent space)
  P(copy | c) = Phi( kappa (1/2 - rho) ),  kappa = ||Delta||^2/sigma_w

python3 -I a2_gibbs_dados.py <coverage2> <results> <out>
"""
import sys, os, json, csv
import numpy as np
from scipy.stats import norm, spearmanr
from scipy.special import logsumexp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_lib import load_latents, load_results, Geometry, RUNS_WIDE

cov, resd, out = sys.argv[1:4]
G = Geometry(load_latents(f'{cov}/{RUNS_WIDE[128]}/step_2500/latents.npz'))
spk, X = G.refs('external')
idx = {s: i for i, s in enumerate(G.spk_ids)}
CB, OBS = {}, {}
for N, r in RUNS_WIDE.items():
    rows = list(csv.DictReader(open(f'{cov}/{r}/step_2500/train_manifest.csv')))
    ss = sorted(set(x['speaker'] for x in rows) - {'lj'})
    CB[N] = np.array([G.means[idx[s]] for s in ss if s in idx])
    res = load_results(f'{resd}/{r}/step_2500/strength_1.0/results.csv')
    OBS[N] = np.array([res[('external', s)]['copy'] for s in spk])
K = 32
SIGW = 0.036
DELTA2 = 0.65   # typical ||Delta||^2 in the metric space (02_derivacao, cos ~0.68); only affects the smoothness
KAPPA = DELTA2 / SIGW


def rho_of(X, cb, T, lnM, v0=G.lj):
    V = np.vstack([v0[None], cb])
    lw = np.r_[lnM, np.full(len(cb), np.log(K))]
    d2 = ((X[:, None, :] - V[None]) ** 2).sum(-1)
    L = lw[None] - d2 / (2 * T)
    P = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    Y = P @ V
    D = X - v0
    rho = 1 - ((Y - v0) * D).sum(1) / (D * D).sum(1)
    return rho, P[:, 0]


def pcopy(X, cb, T, lnM):
    rho, p0 = rho_of(X, cb, T, lnM)
    return norm.cdf(KAPPA * (0.5 - rho)), rho, p0


Ts = np.exp(np.linspace(np.log(0.5), np.log(200), 70))
lnMs = np.linspace(-2, 60, 156)
lnM_nom = np.log(2000)
rows = []
for T in Ts:
    for lnM in lnMs:
        r = dict(T=T, lnM=lnM)
        for N in (8, 32, 128):
            p, rho, p0 = pcopy(X, CB[N], T, lnM)
            r[f'q{N}'] = 1 - p.mean()
            r[f'sse{N}'] = ((p - OBS[N]) ** 2).sum()
            r[f'sp{N}'] = spearmanr(p, OBS[N])[0] if p.std() > 1e-9 else np.nan
        rows.append(r)
import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(f'{out}/a2_grade.csv', index=False)
rep = {'q_obs': {N: float(1 - OBS[N].mean()) for N in OBS}}
# (a) zero parameters: T=1 (Bayes), nominal M
def at(T, lnM):
    o = {}
    for N in (8, 32, 128):
        p, rho, p0 = pcopy(X, CB[N], T, lnM)
        o[N] = dict(q=float(1 - p.mean()), sp=float(spearmanr(p, OBS[N])[0]) if p.std() > 1e-9 else None,
                    rmse_voice=float(np.sqrt(((p - OBS[N]) ** 2).mean())))
    return o
rep['a_T1_Mnom'] = at(1.0, lnM_nom)
# (b) nominal M, T fitted only at N=128 (per-voice sse)
sub = df[np.isclose(df.lnM, lnMs[np.argmin(abs(lnMs - lnM_nom))])]
b = sub.loc[sub.sse128.idxmin()]
rep['b_fitT_on128_Mnom'] = dict(T=float(b['T']), lnM=float(b.lnM), pred=at(b['T'], b.lnM))
# (c) T and lnM fitted only at N=128
c = df.loc[df.sse128.idxmin()]
rep['c_fitTM_on128'] = dict(T=float(c['T']), lnM=float(c.lnM), pred=at(c['T'], c.lnM))
# (d) T and lnM fitted only at N=8, predicting 32 and 128
d = df.loc[df.sse8.idxmin()]
rep['d_fitTM_on8'] = dict(T=float(d['T']), lnM=float(d.lnM), pred=at(d['T'], d.lnM))
# (e) joint fit on all 3 (description, not a test)
df['sse_all'] = df.sse8 + df.sse32 + df.sse128
e = df.loc[df.sse_all.idxmin()]
rep['e_fit_all'] = dict(T=float(e['T']), lnM=float(e.lnM), pred=at(e['T'], e.lnM))
# profile: for each T, the lnM that reproduces q128; what it predicts at 8
prof = []
for T in Ts[::5]:
    s = df[np.isclose(df['T'], T)]
    k = (s.q128 - rep['q_obs'][128]).abs().idxmin()
    prof.append(dict(T=float(T), lnM=float(s.loc[k].lnM), q8=float(s.loc[k].q8), q32=float(s.loc[k].q32),
                     q128=float(s.loc[k].q128), sp8=float(s.loc[k].sp8), sp128=float(s.loc[k].sp128)))
rep['perfil_q128_fixo'] = prof
json.dump(rep, open(f'{out}/a2.json', 'w'), indent=1, default=float)
print(json.dumps(rep, indent=1, default=float))
