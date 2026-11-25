"""b3 (points 5b-bis, 7, 9): (i) mass regime T~20 (per-voice width) against trained voices and the curve in N;
(ii) predicted exponent for continuous measures (sim_ref/sim_spk) via the output-reference distance;
(iii) extrapolation in N under three models of P_v (resampled empirical, Gaussian, Gaussian mixture) with CI;
(iv) local exponent of q(N) from 8 to 1e6 in the Gaussian model (regime crossover).
python3 -I b3_regimes_extrap.py <coverage2> <results> <out>
"""
import sys, os, json, csv
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_lib import load_latents, load_results, Geometry, RUNS_WIDE
from rd_theory import gchisq_cdf_imhof

cov, resd, out = sys.argv[1:4]
rng = np.random.default_rng(33)
KAPPA = 0.65 / 0.036
G = Geometry(load_latents(f'{cov}/{RUNS_WIDE[128]}/step_2500/latents.npz'))
idx = {s: i for i, s in enumerate(G.spk_ids)}
v0, mu = G.lj, G.center
man = lambda run: sorted(set(x['speaker'] for x in csv.DictReader(open(f'{cov}/{run}/step_2500/train_manifest.csv'))) - {'lj'})
spk, X = G.refs('external'); ss, Xs = G.refs('seen')
CB = {N: np.array([G.means[idx[s]] for s in man(r) if s in idx]) for N, r in RUNS_WIDE.items()}
RES = {N: load_results(f'{resd}/{r}/step_2500/strength_1.0/results.csv') for N, r in RUNS_WIDE.items()}
OBS = {N: np.array([RES[N][('external', s)]['copy'] for s in spk]) for N in RUNS_WIDE}
qobs = {N: float(1 - OBS[N].mean()) for N in OBS}
rep = {}


def gibbs(X, cb, T, tau, K=32, gamma=1.0, Ntot=None):
    """mass per codeword K * N^(gamma-1) (total mass K N^gamma)."""
    N = len(cb) if Ntot is None else Ntot
    V = np.vstack([v0[None], cb])
    lw = np.r_[np.log(K) + tau / (2 * T), np.full(len(cb), np.log(K) + (gamma - 1) * np.log(N))]
    d2 = ((X[:, None, :] - V[None]) ** 2).sum(-1)
    L = lw[None] - d2 / (2 * T)
    P = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    Y = P @ V
    D = X - v0
    rho = 1 - ((Y - v0) * D).sum(1) / (D * D).sum(1)
    return norm.cdf(KAPPA * (0.5 - rho)), Y


# (i) mass regime
mass = {}
for T in (1.0, 5.0, 10.0, 21.0):
    for gamma in (1.0, 0.5, 0.25):
        f = lambda t: (1 - gibbs(X, CB[128], T, t, gamma=gamma)[0].mean()) - qobs[128]
        try:
            tau = brentq(f, -400, 1500)
        except ValueError:
            continue
        r = dict(tau=tau, Mef=float(32 * 128 ** gamma / 128 * np.exp(tau / (2 * T))) if gamma == 1 else None)
        for N in (8, 32):
            r[f'q{N}'] = float(1 - gibbs(X, CB[N], T, tau, gamma=gamma)[0].mean())
        # trained voices, N=8, K=8/32/128 (the mass of the voice's own codeword changes with K)
        for K in (8, 32, 128):
            r[f'seen_K{K}'] = float(1 - gibbs(Xs, CB[8], T, tau - 2 * T * np.log(K / 32), K=K, gamma=gamma)[0].mean())
        r['Mhalf_dq128'] = float((1 - gibbs(X, CB[128], T, tau - 2 * T * np.log(2), gamma=gamma)[0].mean()) - qobs[128])
        mass[f'T={T},gamma={gamma}'] = r
        print('mass', T, gamma, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
rep['regime_massa'] = mass
rep['seen_obs_N8'] = {8: 0.5525, 32: 0.26, 128: 0.1075}

# (ii) continuous measure: E||y - c||^2 (Gibbs output vs reference), T=1 and T=21
cont = {}
for T, tau in ((1.0, None), (21.0, None)):
    f = lambda t: (1 - gibbs(X, CB[128], T, t)[0].mean()) - qobs[128]
    tau = brentq(f, -400, 1500)
    y = []
    for N in (8, 32, 128):
        _, Y = gibbs(X, CB[N], T, tau)
        y.append(((Y - X) ** 2).sum(1))
    y = np.array(y)
    a_g = lambda yy: np.log(max((yy[0] - yy[1]), 1e-9) / max((yy[1] - yy[2]), 1e-9)) / np.log(4)
    bs = []
    for b in range(1000):
        ii = rng.integers(0, 38, 38)
        m = y[:, ii].mean(1)
        if m[0] > m[1] > m[2]:
            bs.append(a_g(m))
    m = y.mean(1)
    cont[T] = dict(mean_dist2=m.tolist(), alpha_g=float(a_g(m)), ci90=np.percentile(bs, [5, 95]).tolist(),
                   frac_monotone=len(bs) / 1000)
    # also only the copied-voices part (D_N), without the binary step
    dn = np.array([((X[:, None] - CB[N][None]) ** 2).sum(-1).min(1) for N in (8, 32, 128)])
    cont[f'DN_only_alpha_g'] = float(a_g(dn.mean(1)))
rep['continua'] = cont
print('continuous', cont, flush=True)

# (iii) extrapolation with three models of P_v; tau from the hard rule (real q128)
dLJ2 = ((X - v0) ** 2).sum(1)
D128 = ((X[:, None] - CB[128][None]) ** 2).sum(-1).min(1)
sl = np.sort(dLJ2 - D128); nf = int(round(qobs[128] * 38)); tau_h = (sl[nf - 1] + sl[nf]) / 2
ev, U = np.linalg.eigh(G.Sb); ev = np.clip(ev, 0, None)
noise_extra = np.trace(G.Wx) / 32 / 8 + 1 / 32
from sklearn.mixture import GaussianMixture
means = G.means
bics = {k: GaussianMixture(k, covariance_type='full', reg_covar=0.05, random_state=0, n_init=3).fit(means).bic(means) for k in (1, 2, 3, 4)}
kbest = int(min(bics, key=bics.get))
kmix = max(kbest, 2)
gm = GaussianMixture(kmix, covariance_type='full', reg_covar=0.05, random_state=0, n_init=5).fit(means)
# components: subtract the estimation noise of the means (W_x/8) from the fitted covariance
comps = []
for k in range(kmix):
    C = gm.covariances_[k] - G.Wx / 8
    e_, U_ = np.linalg.eigh(C); e_ = np.clip(e_, 0, None)
    comps.append((gm.weights_[k], gm.means_[k], e_, U_))


def F_gauss(c, t):
    w = ev + noise_extra
    return gchisq_cdf_imhof(t, w, (U.T @ (c - mu)) ** 2 / w) if t > 0 else 0.0


def F_mix(c, t):
    if t <= 0:
        return 0.0
    s = 0.0
    for wk, mk, ek, Uk in comps:
        w = ek + noise_extra
        s += wk * gchisq_cdf_imhof(t, w, (Uk.T @ (c - mk)) ** 2 / w)
    return s


def F_emp(c, t):
    return float(np.mean(((means - c) ** 2).sum(1) <= t))


Ns = [8, 32, 128, 512, 2456, 10000]


def curve(pfun, Xc, tau):
    t = ((Xc - v0) ** 2).sum(1) - tau
    p = np.array([pfun(c, ti) for c, ti in zip(Xc, t)])
    return np.array([np.mean(np.exp(N * np.log1p(-np.minimum(p, 1 - 1e-15)))) for N in Ns]), float(np.mean(t <= 0)), p


ext = {}
for name, pf in (('empirico', F_emp), ('gaussiano', F_gauss), ('mistura', F_mix)):
    qN, fl, p = curve(pf, X, tau_h)
    # anchored: q_obs(128) + (q_model(N) - q_model(128))
    anch = qobs[128] + qN - qN[2]
    bs = []
    for b in range(60):
        ii = rng.integers(0, 38, 38)
        slb = np.sort(dLJ2[ii] - D128[ii]); qb = 1 - OBS[128][ii].mean(); nfb = min(max(int(round(qb * 38)), 1), 37)
        tb = (slb[nfb - 1] + slb[nfb]) / 2
        qb_N, flb, _ = curve(pf, X[ii], tb)
        bs.append(np.r_[qb + qb_N - qb_N[2], flb])
    bs = np.array(bs)
    ext[name] = dict(raw=dict(zip(map(str, Ns), qN.round(4).tolist())), anchored=dict(zip(map(str, Ns), anch.round(4).tolist())),
                     floor=fl, ci90_lo=dict(zip(map(str, Ns + ['floor']), np.percentile(bs, 5, axis=0).round(3).tolist())),
                     ci90_hi=dict(zip(map(str, Ns + ['floor']), np.percentile(bs, 95, axis=0).round(3).tolist())),
                     note_emp='empirical: p = fraction of the 127 voices (resolution 1/127; tail not represented)' if name == 'empirico' else '')
    print(name, ext[name], flush=True)
rep['extrapolacao'] = ext
rep['mistura'] = dict(bic=bics, k=kmix, weights=gm.weights_.round(3).tolist())
# real DN vs mixture (as in a3 for the Gaussian)
dn = {}
for N in (8, 32, 128):
    sims = []
    for r in range(100):
        lab = rng.choice(kmix, N, p=gm.weights_)
        cb = np.array([comps[l][1] + comps[l][3] @ (np.sqrt(comps[l][2] + noise_extra) * rng.standard_normal(32)) for l in lab])
        sims.append(np.median(((X[:, None] - cb[None]) ** 2).sum(-1).min(1)))
    dn[N] = [float(np.mean(sims)), float(np.std(sims))]
rep['DN_mistura'] = dn
print('DN mixture', dn)

# (iv) local exponent of q - q_inf, Gaussian model, N from 8 to 1e6
NsL = [8, 32, 128, 512, 2048, 8192, 32768, 131072, 1048576]
t = ((X - v0) ** 2).sum(1) - tau_h
p = np.array([F_gauss(c, ti) for c, ti in zip(X, t)])
qL = np.array([np.mean(np.exp(N * np.log1p(-np.minimum(p, 1 - 1e-15)))) for N in NsL])
qinf = float(np.mean(p == 0))
loc = [float(-np.log((qL[i + 1] - qinf) / (qL[i] - qinf)) / np.log(NsL[i + 1] / NsL[i])) for i in range(len(NsL) - 1)]
rep['expoente_local_gauss'] = dict(N=NsL, q=qL.round(4).tolist(), qinf=qinf, local_excess=loc,
                                   local_raw=[float(-np.log(qL[i + 1] / qL[i]) / np.log(NsL[i + 1] / NsL[i])) for i in range(len(NsL) - 1)])
print(rep['expoente_local_gauss'])
json.dump(rep, open(f'{out}/b3.json', 'w'), indent=1, default=float)
