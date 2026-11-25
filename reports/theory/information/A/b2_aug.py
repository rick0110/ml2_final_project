"""b2 (ponto 6): pseudo-locutores por velocidade (runs -aug). Onde caem no espaco latente do proprio run?
Codewords reais vs reais+pseudo (LibriTTS) vs pseudo da LJ; efeito na distorcao D_N e na folga das referencias.
python3 -I b2_aug.py <coverage2> <results> <out>
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_lib import load_latents, load_results

cov, resd, out = sys.argv[1:4]
rep = {}
for run in ('n8_k8_s1-aug', 'n8_k128_s1-aug'):
    lat = load_latents(f'{cov}/{run}/step_2500/latents.npz')
    mu, src, spk = lat['mu'].astype(float), lat['source'], lat['speaker']
    tr = src != 'refs'
    real = tr & np.array([('_v' not in s) and s != 'lj' for s in spk])
    # branqueamento pelo W dos locutores REAIS + ruido do posterior
    ids = sorted(set(spk[real]))
    res_ = np.concatenate([mu[real & (spk == s)] - mu[real & (spk == s)].mean(0) for s in ids])
    W = res_.T @ res_ / (len(res_) - len(ids)) + np.diag(np.exp(lat['logvar'][tr]).mean(0))
    e, U = np.linalg.eigh(W); Tm = U @ np.diag(e ** -0.5) @ U.T
    x = mu @ Tm
    groups = sorted(set(spk[tr]))
    M = {s: x[tr & (spk == s)].mean(0) for s in groups}
    n = {s: int((tr & (spk == s)).sum()) for s in groups}
    v0 = M['lj']
    realv = [s for s in groups if '_v' not in s and s != 'lj']
    pseu = [s for s in groups if '_v' in s and not s.startswith('lj')]
    ljp = [s for s in groups if s.startswith('lj_v')]
    d = lambda a, b: float(np.linalg.norm(M[a] - M[b]))
    src_d = [d(p, p.split('_v')[0]) for p in pseu if p.split('_v')[0] in M]
    other_d = []
    for p in pseu:
        s0 = p.split('_v')[0]
        other_d.append(min(d(p, r) for r in realv if r != s0))
    real_nn = [min(d(a, b) for b in realv if b != a) for a in realv]
    ljp_d = {p: d(p, 'lj') for p in ljp}
    out_ = dict(n_real=len(realv), n_pseudo=len(pseu), n_ljpseudo=len(ljp),
                pseudo_to_source_median=float(np.median(src_d)), pseudo_to_other_real_median=float(np.median(other_d)),
                real_to_nearest_real_median=float(np.median(real_nn)), lj_pseudo_to_lj=ljp_d,
                lj_to_real_center=float(np.linalg.norm(v0 - np.mean([M[s] for s in realv], 0))),
                utts_per_pseudo_median=float(np.median([n[p] for p in pseu])))
    # projecao no eixo de tom: os pseudo se movem ao longo de uma direcao comum?
    shifts = np.array([M[p] - M[p.split('_v')[0]] for p in pseu if p.split('_v')[0] in M])
    sv = np.linalg.svd(shifts - 0, compute_uv=False)
    out_['shift_PR'] = float(sv.sum() ** 2 / (sv ** 2).sum()) if len(sv) else None
    out_['shift_top1_frac'] = float(sv[0] ** 2 / (sv ** 2).sum())
    # referencias
    rr = load_results(f'{resd}/{run}/step_2500/strength_1.0/results.csv')
    refs = {}
    for kind in ('unseen', 'seen'):
        m = (src == 'refs') & (lat['emotion'] == kind)
        for s, c in zip(spk[m], x[m]):
            Dr = min(((c - M[r]) ** 2).sum() for r in realv)
            Dp = min(((c - M[r]) ** 2).sum() for r in realv + pseu)
            Dlj = min(((c - M[r]) ** 2).sum() for r in ljp)
            refs[f'{kind}:{s}'] = dict(dLJ2=float(((c - v0) ** 2).sum()), D_real=float(Dr), D_real_pseudo=float(Dp),
                                       D_lj_pseudo=float(Dlj), slack_real=float(((c - v0) ** 2).sum() - Dr),
                                       slack_aug=float(min(((c - v0) ** 2).sum(), Dlj) - Dp),
                                       copy=rr[(kind, s)]['copy'])
    out_['refs'] = refs
    rep[run] = out_
    print(run, json.dumps({k: v for k, v in out_.items() if k != 'refs'}, default=float))
    for k, v in refs.items():
        print('  ', k, {a: round(b, 2) for a, b in v.items()})
json.dump(rep, open(f'{out}/b2.json', 'w'), indent=1, default=float)
