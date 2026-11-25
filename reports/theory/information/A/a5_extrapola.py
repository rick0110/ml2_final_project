"""a5: semi-parametric model (empirical codebook + Gaussian tail), extrapolation in N,
sample complexity in bits, effects of K and M, floor of a voice module of dimension k, and
the case without a dominant voice (two stages). Hard rule (limit T->0):
    copied <=> D_N(c) < t(c) = ||c - v0||^2 - tau.
python3 -I a5_extrapola.py <coverage2> <results> <out>
"""
import sys, os, json, csv
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_lib import load_latents, load_results, Geometry, RUNS_WIDE
from rd_theory import gchisq_cdf_imhof, waterfill

cov, resd, out = sys.argv[1:4]
rng = np.random.default_rng(3)
G = Geometry(load_latents(f'{cov}/{RUNS_WIDE[128]}/step_2500/latents.npz'))
idx = {s: i for i, s in enumerate(G.spk_ids)}
spk, X = G.refs('external')
ev, U = np.linalg.eigh(G.Sb); ev = np.clip(ev, 0, None)
order = np.argsort(ev)[::-1]; ev, U = ev[order], U[:, order]
mu, v0, Wx = G.center, G.lj, G.Wx
CB128 = G.means  # 127 trained voices (means of ~8 utterances)
nb = len(CB128)
res = {N: load_results(f'{resd}/{r}/step_2500/strength_1.0/results.csv') for N, r in RUNS_WIDE.items()}
obs = {N: np.array([res[N][('external', s)]['copy'] for s in spk]) for N in RUNS_WIDE}
qobs = {N: float(1 - obs[N].mean()) for N in obs}
K = 32
EXTRA = np.trace(Wx) / 32 / 8   # estimation noise of the means (~8 utterances) per direction


def codebook(run):
    rows = list(csv.DictReader(open(f'{cov}/{run}/step_2500/train_manifest.csv')))
    ss = sorted(set(x['speaker'] for x in rows) - {'lj'})
    return np.array([G.means[idx[s]] for s in ss if s in idx])


dLJ2 = ((X - v0) ** 2).sum(1)
D128 = ((X[:, None] - codebook(RUNS_WIDE[128])[None]) ** 2).sum(-1).min(1)
slack = dLJ2 - D128
s_sorted = np.sort(slack)
nf = int(round(qobs[128] * len(X)))
tau = float((s_sorted[nf - 1] + s_sorted[nf]) / 2)   # hard rule: q128 = 8/38
rep = dict(tau_hard=tau, q_obs=qobs)


def p_emp_gauss(Xc, tau, Kk=K, dK=0.0):
    """per-voice small-ball probability: empirical (127 voices) when there are >=3 neighbours within the radius,
    otherwise Gaussian (Imhof) x a calibration factor estimated on the voices with >=3 neighbours."""
    t = ((Xc - v0) ** 2).sum(1) - tau - dK
    d2 = ((Xc[:, None] - CB128[None]) ** 2).sum(-1)
    kc = (d2 <= t[:, None]).sum(1)
    w = ev + 1.0 / Kk + EXTRA
    pg = np.array([gchisq_cdf_imhof(ti, w, (U.T @ (c - mu)) ** 2 / w) if ti > 0 else 0.0 for c, ti in zip(Xc, t)])
    return t, kc, pg


def q_of_N(p, Ns):
    return np.array([np.mean(np.exp(N * np.log1p(-np.minimum(p, 1 - 1e-15)))) for N in Ns])


Ns = [1, 2, 4, 8, 16, 32, 64, 128, 247, 512, 1000, 2048, 2456, 5000, 10000, 100000]
t, kc, pg = p_emp_gauss(X, tau)
m = kc >= 3
ratio = float(np.exp(np.median(np.log((kc[m] / nb) / np.maximum(pg[m], 1e-300)))))
rep['calib_ratio_emp_over_gauss'] = ratio
rep['n_voices_kc>=3'] = int(m.sum())
p = np.where(kc >= 3, kc / nb, np.minimum(ratio * pg, 1.0))
qN = q_of_N(p, Ns)
rep['q_semipar'] = dict(zip(map(str, Ns), qN.round(4).tolist()))
rep['floor_t<=0'] = float(np.mean(t <= 0))
# random subsets (without replacement) of the 127 for N<=127: empirical expected curve
sub = {}
for N in (8, 16, 32, 64, 127):
    qs = []
    for r in range(300):
        cbs = CB128[rng.choice(nb, N, replace=False)]
        D = ((X[:, None] - cbs[None]) ** 2).sum(-1).min(1)
        qs.append(np.mean(D > t))
    sub[N] = [float(np.mean(qs)), float(np.std(qs))]
rep['q_random_subsets_of_127'] = sub
# per-voice code length (bits) and quantiles -> N90, N95
with np.errstate(divide='ignore'):
    L = -np.log2(p)
rep['L_bits_quantiles'] = {str(qq): float(np.quantile(L, qq)) for qq in (0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95)}


def Nfor(target_q, p):
    f = lambda lnN: np.mean(np.exp(np.exp(lnN) * np.log1p(-np.minimum(p, 1 - 1e-15)))) - target_q
    if f(np.log(1e12)) > 0:
        return np.inf
    return float(np.exp(brentq(f, 0, np.log(1e12))))
rep['N90'] = Nfor(0.10, p); rep['N95'] = Nfor(0.05, p)
# bootstrap over voices (refits tau in each resample: calibration + sampling uncertainty)
boots = []
for b in range(400):
    ii = rng.integers(0, len(X), len(X))
    sl = np.sort(slack[ii]); qb = 1 - obs[128][ii].mean(); nfb = int(round(qb * len(ii)))
    nfb = min(max(nfb, 1), len(ii) - 1)
    taub = (sl[nfb - 1] + sl[nfb]) / 2
    tb = dLJ2[ii] - taub
    d2 = ((X[ii][:, None] - CB128[None]) ** 2).sum(-1)
    kb = (d2 <= tb[:, None]).sum(1)
    # rescaled Gaussian: reuse pg approximating with interpolation in t (cost) -> recompute only if t changed
    w = ev + 1.0 / K + EXTRA
    pgb = np.array([gchisq_cdf_imhof(ti, w, (U.T @ (c - mu)) ** 2 / w) if ti > 0 else 0.0
                    for c, ti in zip(X[ii], tb)]) if b < 120 else None
    if pgb is None:
        continue
    pb = np.where(kb >= 3, kb / nb, np.minimum(ratio * pgb, 1.0))
    boots.append(np.r_[q_of_N(pb, [8, 32, 512, 2456, 10000]), Nfor(0.1, pb), np.mean(tb <= 0)])
boots = np.array(boots)
rep['boot'] = dict(cols=['q8', 'q32', 'q512', 'q2456', 'q1e4', 'N90', 'floor'],
                   p05=np.nanpercentile(np.where(np.isinf(boots), 1e12, boots), 5, axis=0).round(4).tolist(),
                   p50=np.nanpercentile(np.where(np.isinf(boots), 1e12, boots), 50, axis=0).round(4).tolist(),
                   p95=np.nanpercentile(np.where(np.isinf(boots), 1e12, boots), 95, axis=0).round(4).tolist())
# effects of K and M (T in {1,3}): tau -> tau + 2T ln((M'/M)(K/K')), codeword noise 32(1/K'-1/K)
var = {}
for T in (1.0, 3.0):
    for lab, Kp, Mf in (('K8', 8, 1), ('K128', 128, 1), ('Mhalf', 32, 0.5), ('M0.1', 32, 0.1)):
        dtau = 2 * T * np.log(Mf * Kp / K)   # tau' = tau - dtau... (tau = 2T ln(M/K))
        taup = tau + 2 * T * np.log(Mf) - 2 * T * np.log(Kp / K)
        tt, kk, pp = p_emp_gauss(X, taup, Kk=Kp, dK=32 * (1 / Kp - 1 / K))
        pq = np.where(kk >= 3, kk / nb, np.minimum(ratio * pp, 1.0))
        var[f'{lab}_T{int(T)}'] = dict(zip(['8', '32', '128', '2456'], q_of_N(pq, [8, 32, 128, 2456]).round(4).tolist()))
rep['variacoes'] = var
# voice module of dimension k (subspace of the k largest directions of Sb)
mod = {}
for k in (1, 2, 3, 4, 6, 8, 12, 16, 32):
    P = U[:, :k] @ U[:, :k].T
    Q = np.eye(32) - P
    tail = ((X - mu) @ Q.T)
    Dinf = (tail ** 2).sum(1)  # perfect codebook in the subspace
    row = dict(floor=float(np.mean(Dinf > t)), Dtail_mean=float(Dinf.mean()), lam_tail=float(ev[k:].sum()))
    for N, run in RUNS_WIDE.items():
        cb = codebook(run)
        Dk = ((((X - mu) @ P.T)[:, None] - ((cb - mu) @ P.T)[None]) ** 2).sum(-1).min(1) + Dinf
        row[f'q{N}'] = float(np.mean(Dk > t))
    mod[k] = row
rep['modulo_k'] = mod
# no dominant voice: v0 -> mu, tau -> 0 (copy <=> codebook beats the zero-rate code)
t0 = ((X - mu) ** 2).sum(1)
d2 = ((X[:, None] - CB128[None]) ** 2).sum(-1)
nd = {}
for N, run in RUNS_WIDE.items():
    cb = codebook(run)
    nd[N] = float(np.mean(((X[:, None] - cb[None]) ** 2).sum(-1).min(1) > t0))
kc0 = (d2 <= t0[:, None]).sum(1)
w = ev + 1.0 / K + EXTRA
pg0 = np.array([gchisq_cdf_imhof(ti, w, (U.T @ (c - mu)) ** 2 / w) for c, ti in zip(X, t0)])
p0 = np.where(kc0 >= 3, kc0 / nb, np.minimum(ratio * pg0, 1))
nd['semipar'] = dict(zip(map(str, [8, 32, 128, 512, 2456, 10000]), q_of_N(p0, [8, 32, 128, 512, 2456, 10000]).round(4).tolist()))
rep['sem_voz_dominante'] = nd
# bits: voice information in one utterance and reverse water-filling rate at each voice's tolerance
rep['I_voice_one_utt_bits'] = float(0.5 * np.sum(np.log2(1 + ev)))
rep['I_voice_mean_K32_bits'] = float(0.5 * np.sum(np.log2(1 + ev * 32)))
noise = np.trace(Wx) + 32 / K
Rwf = []
for ti in t:
    tv = ti - noise
    Rwf.append(0.0 if tv >= ev.sum() else (np.inf if tv <= 0 else
               brentq(lambda R: waterfill(ev[ev > 1e-6], R)[0] - tv, 0, 300)))
Rwf = np.array(Rwf)
rep["Rwf_bits_quantiles"] = {str(qq): float(np.quantile(np.where(np.isinf(Rwf), 1e9, Rwf), qq)) for qq in (0.5, 0.75, 0.8, 0.9, 0.95)}
rep['frac_t_below_noise'] = float(np.mean(t <= noise))
rep['noise_floor_trace'] = float(noise)
json.dump(rep, open(f'{out}/a5.json', 'w'), indent=1, default=float)
print(json.dumps(rep, indent=1, default=float))
