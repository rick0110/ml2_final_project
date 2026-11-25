"""t1: rate-distortion for a Gaussian source with a power-law spectrum (part 1 of the report).
Compares, for R = log2 N:
  (i)  D_wf(R): reverse water-filling (optimal codebook; one-shot lower bound: E d >= D(log2 N))
  (ii) D_rand(N): random codebook, N iid codewords from the source itself (exact Monte Carlo)
  (iii) saddle-point / Chernoff approximation for the distribution of D_N
  (iv) high resolution (nearest neighbour): Gamma(1+2/d) (N V_d)^(-2/d) E[f^(-2/d)]
and checks Imhof vs saddle point vs Monte Carlo for the small-ball probability.
python3 -I t1_rd_teoria.py <out_dir> <Sb_eig.npy>
"""
import sys, os, json
import numpy as np
from scipy.special import gammaln
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_theory import gchisq_cdf_imhof, smallball_saddle, waterfill

out, eigf = sys.argv[1:3]
rng = np.random.default_rng(0)
lam_meas = np.load(eigf)
lam_meas = lam_meas[lam_meas > 1e-6]
tr = lam_meas.sum()
d = 32
spectra = {}
for a in (1.0, 1.5, 2.0):
    l = np.arange(1, d + 1) ** (-a); spectra[f'a={a}'] = l / l.sum() * tr
spectra['medido'] = np.r_[lam_meas, np.full(d - len(lam_meas), 1e-6)][:d] if len(lam_meas) < d else lam_meas[:d]
Ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
rep = {}
for name, lam in spectra.items():
    lam = np.sort(lam)[::-1]
    row = {}
    dwf = [waterfill(lam, np.log2(N))[0] for N in Ns]
    # MC random codebook
    nrep = 400
    X = rng.standard_normal((nrep, d)) * np.sqrt(lam)
    drand = []
    for N in Ns:
        vals = []
        for i in range(nrep):
            Y = rng.standard_normal((N, d)) * np.sqrt(lam)
            vals.append(((Y - X[i]) ** 2).sum(1).min())
        drand.append(float(np.mean(vals)))
    # high resolution: d_eff = number of dims with lam > 1e-4*lam1
    lam_hr = lam[lam > 1e-6]
    de = len(lam_hr)
    lnVd = (de / 2) * np.log(np.pi) - gammaln(de / 2 + 1)
    lnEf = np.log(2 * np.pi) + np.sum(np.log(lam_hr)) / de - (de / 2) * np.log(1 - 2 / de)
    dhr = [float(np.exp(gammaln(1 + 2 / de) - (2 / de) * (np.log(N) + lnVd) + lnEf)) for N in Ns]
    # saddle point: median D_N for a typical c: solve N p(t) = ln 2, mean over 60 c's
    dsad = []
    for N in Ns:
        meds = []
        for i in range(60):
            a2 = X[i] ** 2 / lam
            lo, hi = 1e-6, 4 * tr
            for _ in range(50):
                mid = np.sqrt(lo * hi)
                p = smallball_saddle(mid, lam, a2)
                if N * p > np.log(2): hi = mid
                else: lo = mid
            meds.append(mid)
        dsad.append(float(np.mean(meds)))
    row['N'] = Ns; row['D_wf'] = [float(x) for x in dwf]; row['D_rand_MC_mean'] = drand
    row['D_rand_saddle_median'] = dsad; row['D_highres'] = dhr; row['trace'] = float(lam.sum())
    sl = lambda D: [float(-(np.log(D[i + 1]) - np.log(D[i])) / (np.log(Ns[i + 1]) - np.log(Ns[i]))) for i in range(len(D) - 1)]
    row['slope_wf'] = sl(dwf); row['slope_rand'] = sl(drand)
    row['ratio_rand_wf'] = [float(a / b) for a, b in zip(drand, dwf)]
    # extra rate of the random codebook: R_rand(D) - R_wf(D) in bits, at D = D_rand(N)
    extra = []
    for N, Dr in zip(Ns, drand):
        lo, hi = 0.0, 200.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if waterfill(lam, mid)[0] > Dr: lo = mid
            else: hi = mid
        extra.append(float(np.log2(N) - mid))
    row['rate_penalty_bits'] = extra
    rep[name] = row
    print(name, 'D_wf', np.round(dwf, 2), '\n  D_rand', np.round(drand, 2), '\n  saddle', np.round(dsad, 2),
          '\n  highres', np.round(dhr, 2), '\n  slope_rand', np.round(row['slope_rand'], 3),
          '\n  slope_wf', np.round(row['slope_wf'], 3), '\n  penalty bits', np.round(extra, 2), flush=True)

# Imhof / saddle / MC check for small balls (measured spectrum)
lam = np.sort(spectra['medido'])[::-1]
chk = []
for i in range(4):
    c = rng.standard_normal(d) * np.sqrt(lam) * 1.3
    a2 = c ** 2 / lam
    M = 400000
    Y = rng.standard_normal((M, d)) * np.sqrt(lam)
    dist = ((Y - c) ** 2).sum(1)
    for qq in (0.2, 0.01, 0.001):
        t = np.quantile(dist, qq)
        chk.append(dict(c=i, t=float(t), MC=float((dist <= t).mean()), imhof=gchisq_cdf_imhof(t, lam, a2),
                        saddle=smallball_saddle(t, lam, a2), chernoff_bound=float(np.exp(-smallball_saddle(t, lam, a2, True)[1]))))
rep['check_smallball'] = chk
for r in chk: print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
json.dump(rep, open(f'{out}/t1.json', 'w'), indent=1)
