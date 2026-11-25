"""t3: checks the law q(N) - q_inf ~ A N^(-2/d) (Tauberian theorem) on a synthetic population:
voices uniform in [0,1]^d (density f=1), per-voice tolerance t(c) with density g(t) > 0 at t=0
(a fraction q_inf with t<=0 is never copied). Failure <=> D_N(c) = min_i ||c-v_i||^2 > t(c).
Also measures the local exponent between N=8 and 128 and the ratio of decrements (q32-q128)/(q8-q32) = 4^(-2/d).
python3 -I t3_tauberiano.py <out>
"""
import sys, json
import numpy as np
out = sys.argv[1]
rng = np.random.default_rng(11)
rep = {}
for d in (4, 8, 12):
    nc = 3000
    C = rng.random((nc, d))
    t = rng.uniform(-0.05, 0.6, nc) * (d / 8)  # arbitrary scale; fraction t<=0 = 0.05/0.65
    qinf = float(np.mean(t <= 0))
    Ns = [8, 32, 128, 512, 2048, 8192]
    qs = []
    for N in Ns:
        f = []
        for r in range(3):
            V = rng.random((N, d))
            # toroidal distance to avoid edge effects
            D = np.full(nc, np.inf)
            for s in range(0, N, 1024):
                diff = np.abs(C[:, None, :] - V[None, s:s + 1024, :]); diff = np.minimum(diff, 1 - diff)
                D = np.minimum(D, (diff ** 2).sum(-1).min(1))
            f.append(np.mean(D > t))
        qs.append(float(np.mean(f)))
    ex = qs and np.polyfit(np.log(Ns[2:]), np.log(np.array(qs[2:]) - qinf), 1)[0]
    rep[d] = dict(q=qs, qinf=qinf, slope_excess_largeN=float(ex), pred=-2 / d,
                  ratio_decr=float((qs[1] - qs[2]) / (qs[0] - qs[1])), ratio_pred=float(4 ** (-2 / d)),
                  local_exp_8_128=float(-np.log(qs[2] / qs[0]) / np.log(16)))
    print(d, {k: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)) for k, v in rep[d].items()}, flush=True)
json.dump(rep, open(f'{out}/t3.json', 'w'), indent=1)
