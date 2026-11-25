"""a1: geometria medida no espaco branqueado do run N=128 (-wide) e calibracao do modelo.
python3 -I a1_geometria.py <coverage2_dir> <results_dir> <out_dir>
Mede: espectro entre-locutores (corrigido de ruido), posicao da voz padrao, ruido da referencia,
distancias das 38 referencias externas a LJ e aos livros-codigo aninhados N=8,32,128."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rd_lib import load_latents, load_results, Geometry, RUNS_WIDE

cov, resd, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
G = Geometry(load_latents(f'{cov}/{RUNS_WIDE[128]}/step_2500/latents.npz'))
ev, U = np.linalg.eigh(G.Sb); ev, U = ev[::-1], U[:, ::-1]
evp = np.clip(ev, 0, None)
rep = {}
rep['Sb_eig'] = ev.round(3).tolist()
rep['tr_Sb'] = float(evp.sum())
rep['PR_Sb'] = float(evp.sum() ** 2 / (evp ** 2).sum())
# ajuste de lei de potencia nas 16 primeiras
j = np.arange(1, 17)
a, b = np.polyfit(np.log(j), np.log(evp[:16]), 1)
rep['alpha_16'] = float(-a); rep['logc_16'] = float(b)
j8 = np.arange(1, 9); a8, _ = np.polyfit(np.log(j8), np.log(evp[:8]), 1); rep['alpha_8'] = float(-a8)
dlj = G.lj - G.center
rep['d_LJ_center'] = float(np.linalg.norm(dlj))
proj = U.T @ dlj
rep['LJ_proj_on_Sb_axes_sq_frac_top'] = (proj[:8] ** 2 / (dlj @ dlj)).round(3).tolist()
rep['LJ_offset_in_Sb_span_frac'] = float((proj[evp > 0.05] ** 2).sum() / (dlj @ dlj))
rep['LJ_offset_mahal_Sb_top8'] = float(np.sqrt((proj[:8] ** 2 / evp[:8]).sum()))
rep['lj_within_trace'] = float(np.trace(G.lj_cov)); rep['libri_within_trace(Wx)'] = float(np.trace(G.Wx))
rep['dim'] = int(G.x.shape[1])
# ruido da referencia: uma frase -> covariancia Wx (mu, sem amostragem); tr
spk, X = G.refs('external')
rep['n_ext'] = int(len(spk))
dL = np.linalg.norm(X - G.lj, axis=1)
dC = np.linalg.norm(X - G.center, axis=1)
rep['ref_dLJ_mean_sd'] = [float(dL.mean()), float(dL.std())]
rep['ref_dCenter_mean_sd'] = [float(dC.mean()), float(dC.std())]
# typical distance between 2 libri voices (noise-free) sqrt(2 tr Sb)
rep['typical_voice_dist'] = float(np.sqrt(2 * evp.sum()))
# codebooks aninhados
import csv
sets = {}
for N, r in RUNS_WIDE.items():
    rows = list(csv.DictReader(open(f'{cov}/{r}/step_2500/train_manifest.csv')))
    sets[N] = sorted(set(x['speaker'] for x in rows) - {'lj'})
idx = {s: i for i, s in enumerate(G.spk_ids)}
res = {N: load_results(f'{resd}/{r}/step_2500/strength_1.0/results.csv') for N, r in RUNS_WIDE.items()}
table = []
for N in (8, 32, 128):
    cb = np.array([G.means[idx[s]] for s in sets[N] if s in idx])
    nn = np.sqrt(((X[:, None, :] - cb[None]) ** 2).sum(-1)).min(1)
    cop = np.array([res[N][('external', s)]['copy'] for s in spk])
    f0 = np.array([res[N][('external', s)]['f0'] for s in spk])
    from scipy.stats import spearmanr
    gap = dL ** 2 - nn ** 2
    row = dict(N=N, n_codebook_present=int(len(cb)), q_obs=float(1 - cop.mean()),
               dNN_mean=float(nn.mean()), rho_copy_dLJ=float(spearmanr(cop, dL)[0]),
               rho_copy_dNN=float(spearmanr(cop, nn)[0]), rho_copy_gap=float(spearmanr(cop, gap)[0]),
               rho_copy_f0=float(spearmanr(cop, f0)[0]))
    table.append(row)
    np.savetxt(f'{out}/a1_pervoice_N{N}.csv', np.c_[cop, dL, nn, dC, f0], delimiter=',',
               header='copy,dLJ,dNN,dCenter,f0', comments='')
rep['nested'] = table
np.save(f'{out}/a1_Sb_eig.npy', evp)
json.dump(rep, open(f'{out}/a1.json', 'w'), indent=1)
print(json.dumps(rep, indent=1))
