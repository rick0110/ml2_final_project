"""b22: elos latentes da cadeia de informacao sobre as 8 vozes treinadas comuns: mu e z = mu + sigma*eps.
Identificacao por centroide mais proximo (Euclides branqueado por W), deixando a frase de fora; Fano com 8 classes.
Uso: python3 -I b22_cadeia_latente.py <coverage2_root> out/b0.npz"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bt_lib import load
root = sys.argv[1]; S = load(sys.argv[2]); rng = np.random.default_rng(3)
SEEN = sorted(x["ref"] for x in S["n8_k128_s1-spk"]["refs"] if x["grp"] == "seen"); SPK = [r.split("_")[0] for r in SEEN]
def fano(acc, n):
    pe = min(max(1 - acc, 1e-12), 1 - 1e-12); return np.log2(n) - (-(pe*np.log2(pe)+(1-pe)*np.log2(1-pe))) - pe*np.log2(n-1)
res = {}
for run in ["n8_k32_s1-wide", "n32_k32_s1-wide", "n128_k32_s1-wide", "n8_k8_s1", "n8_k128_s1", "n32_k8_s1", "n32_k128_s1", "n128_k8_s1", "n200_k32_s1"]:
    z = np.load(os.path.join(root, run, "step_2500", "latents.npz"), allow_pickle=False)
    mu = z["mu"].astype(float); sd = np.exp(0.5 * z["logvar"].astype(float)); spk = z["speaker"].astype(str); src = z["source"].astype(str); uid = z["utt_id"].astype(str)
    tr = (src != "refs") & (spk != "lj")
    lab = spk[tr]; X = mu[tr]; Sd = sd[tr]; u = np.unique(lab); Mn = np.array([X[lab == s].mean(0) for s in u])
    R = X - Mn[np.searchsorted(u, lab)]; W = R.T @ R / (len(X) - len(u)); W += 0.05 * np.trace(W) / 32 * np.eye(32); Li = np.linalg.inv(np.linalg.cholesky(W))
    out = {}
    for kind in ["mu", "z"]:
        hits = []; nt = 0
        for j, s in enumerate(SPK):
            idx = np.where(tr & (spk == s))[0]
            for i in idx:
                cents = []
                for s2 in SPK:
                    m = tr & (spk == s2); m[i] = False; cents.append(mu[m].mean(0))
                x = mu[i] + (rng.standard_normal(32) * sd[i] if kind == "z" else 0)
                d = (((np.array(cents) - x) @ Li.T) ** 2).sum(1); hits.append(d.argmin() == j)
        acc = np.mean(hits); out[kind] = dict(acc=float(acc), fano=float(fano(acc, 8)), n=len(hits))
    # the actual reference latents (one per voice)
    hits = []
    for j, (r, s) in enumerate(zip(SEEN, SPK)):
        i = np.where(uid == r)[0]
        if len(i) == 0: continue
        cents = np.array([mu[tr & (spk == s2)].mean(0) for s2 in SPK]); d = (((cents - mu[i[0]]) @ Li.T) ** 2).sum(1); hits.append(d.argmin() == j)
    out["ref_mu_acc"] = float(np.mean(hits)) if hits else None
    res[run] = out
    print(f"{run:18s}: mu acc {out['mu']['acc']:.3f} Fano {out['mu']['fano']:.2f} (n={out['mu']['n']}) | z acc {out['z']['acc']:.3f} Fano {out['z']['fano']:.2f} | latente das 8 referencias: acerto {out['ref_mu_acc']}")
json.dump(res, open("out/b22.json", "w"), indent=1)
