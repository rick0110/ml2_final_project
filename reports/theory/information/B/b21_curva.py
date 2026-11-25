"""b21: curva C(N,K) num conjunto comum de vozes, consistencia s0<->s1, e decomposicao I = H(Y) - H(Y|S).
Uso: python3 -I b21_curva.py out/b0.npz out/b12_emb.npz out/b20_outs.npz [out/b20_refs.npz] [B]"""
import sys, os, re, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr, linregress
from bt_lib import load
S = load(sys.argv[1]); refs_file = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4].endswith(".npz") else None
B = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 1000; rng = np.random.default_rng(12)
def ld(p):
    d = np.load(p, allow_pickle=False); E = d["emb"] / np.linalg.norm(d["emb"], axis=1, keepdims=True); return list(d["keys"].astype(str)), E
k1, E1 = ld(sys.argv[2]); k2, E2 = ld(sys.argv[3]); keys = k1 + k2; E = np.vstack([E1, E2])
OUT = {}
for i, k in enumerate(keys):
    p = k.split("|")
    if p[0] == "out":
        ref, sent = p[3].rsplit("__", 1); OUT.setdefault((p[1], p[2]), {}).setdefault(ref, {})[sent] = E[i]
V12 = sorted(x["ref"] for x in S["n8_k128_s1-spk"]["refs"] if x["grp"] in ("seen", "unseen"))
SEEN = sorted(x["ref"] for x in S["n8_k128_s1-spk"]["refs"] if x["grp"] == "seen"); UNS = sorted(x["ref"] for x in S["n8_k128_s1-spk"]["refs"] if x["grp"] == "unseen")
EXT4 = sorted(x["ref"] for x in S["n128_k32_s1"]["refs"] if x["grp"] == "external")
cent = {}
if refs_file:
    k3, E3 = ld(refs_file)
    for i, k in enumerate(k3):
        p = k.split("|")
        if p[0] == "oth": cent.setdefault(p[1], []).append(E3[i])
for i, k in enumerate(k1):
    p = k.split("|")
    if p[0] == "oth": cent.setdefault(p[1], []).append(E1[i])
CEN = {s: (np.mean(v, 0) / np.linalg.norm(np.mean(v, 0))) for s, v in cent.items()}
def meta(run):
    m = re.match(r"n(\d+)_k(\d+)_s(\d+)(?:-(.*))?$", run); return int(m.group(1)), int(m.group(2)), m.group(4) or "base"
def fano(acc, n):
    pe = min(max(1 - acc, 1e-12), 1 - 1e-12); return np.log2(n) - (-(pe*np.log2(pe)+(1-pe)*np.log2(1-pe))) - pe*np.log2(n-1)
def lovo_rows(sim, true):
    betas = np.linspace(1, 150, 150)
    tab = np.array([[-(b * sim[i, true[i]] - logsumexp(b * sim[i])) / np.log(2) for i in range(len(sim))] for b in betas])
    tot = tab.sum(1); return np.array([tab[np.argmin(tot - tab[:, i]), i] for i in range(len(sim))])
def ident(run, cond, vs):
    """saidas (todas as frases) da voz v identificadas entre centroides reais das vozes vs"""
    o = OUT.get((run, cond), {}); spk = [v.split("_")[0] for v in vs]
    if not all(s in CEN for s in spk) or not all(v in o for v in vs): return None
    C = np.array([CEN[s] for s in spk]); Y, t, vid = [], [], []
    for j, v in enumerate(vs):
        for s_, e in o[v].items(): Y.append(e); t.append(j); vid.append(j)
    Y, t, vid = np.array(Y), np.array(t), np.array(vid); sim = Y @ C.T
    acc = np.mean(sim.argmax(1) == t); rows = lovo_rows(sim, t); n = len(vs)
    bs = []
    for _ in range(B):
        pick = rng.integers(0, n, n); ii = np.concatenate([np.where(vid == p)[0] for p in pick])
        bs.append([fano(np.mean(sim[ii].argmax(1) == t[ii]), n), np.log2(n) - rows[ii].mean()])
    bs = np.array(bs)
    return dict(n=n, trials=len(t), acc=float(acc), fano=float(fano(acc, n)), ce=float(np.log2(n) - rows.mean()),
                fano_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), ce_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist())
def consist(run, cond, vs):
    o = OUT.get((run, cond), {}); vs = [v for v in vs if v in o and "s0" in o[v] and "s1" in o[v]]
    if len(vs) < 4: return None
    A = np.array([o[v]["s0"] for v in vs]); Bm = np.array([o[v]["s1"] for v in vs]); sim = A @ Bm.T; t = np.arange(len(vs))
    return dict(n=len(vs), acc=float(np.mean(sim.argmax(1) == t)), fano=float(fano(np.mean(sim.argmax(1) == t), len(vs))), ce=float(np.log2(len(vs)) - lovo_rows(sim, t).mean()))
def decomp(run, cond, vs):
    """espalhamento (H) e ruido por voz (H|S), escalar: D_tot = dist^2 media entre saidas de vozes diferentes; D_w = dist^2 media s0-s1 da mesma voz / 2"""
    o = OUT.get((run, cond), {}); vs = [v for v in vs if v in o and "s0" in o[v] and "s1" in o[v]]
    if len(vs) < 4: return None
    A = np.array([o[v]["s0"] for v in vs]); Bm = np.array([o[v]["s1"] for v in vs]); M_ = (A + Bm) / 2
    Dw = np.mean(((A - Bm) ** 2).sum(1)) / 2                       # variancia dentro (por amostra, soma das dims)
    Dm = np.mean([((M_[i] - M_[j]) ** 2).sum() for i in range(len(vs)) for j in range(len(vs)) if i < j]) / 2   # variancia das medias (com Dw/2)
    Db = Dm - Dw / 2                                                # variancia entre vozes, sem o ruido
    return dict(n=len(vs), D_within=float(Dw), D_between=float(Db), D_total=float(Db + Dw), snr=float(Db / Dw),
                I_iso_per_dim=float(0.5 * np.log2(1 + Db / Dw)))
runs = sorted(set(r for r, c in OUT), key=lambda r: (meta(r)[0], meta(r)[1], r))
res = {}
print(f"conjunto comum V12 = 8 treinadas + 4 novas (log2 12 = {np.log2(12):.2f}); centroides reais disponiveis: {sum(v.split('_')[0] in CEN for v in V12)}/12; EXT4 {sum(v.split('_')[0] in CEN for v in EXT4)}/4")
print("\n(A) fidelidade (saida x centroide real) e (B) consistencia s0<->s1, (C) decomposicao escalar; forca 1.0, R=1")
hdr = f"{'treino':22s} {'N':>4s} {'K':>4s} | V12 Fano  EC [IC]          | treinadas(8) EC | novas(4) EC | consist. EC | D_within D_between I_iso/dim"
print(hdr)
for r in runs:
    N, K, var = meta(r)
    if (r, "strength_1.0") not in OUT: continue
    a = ident(r, "strength_1.0", V12); s8 = ident(r, "strength_1.0", SEEN); u4 = ident(r, "strength_1.0", UNS)
    c = consist(r, "strength_1.0", V12); dd = decomp(r, "strength_1.0", V12)
    res[r] = dict(N=N, K=K, var=var, V12=a, seen=s8, uns=u4, cons=c, dec=dd)
    f = lambda x, key="ce": f"{x[key]:.2f}" if x else "  — "
    line = f"{r:22s} {N:4d} {K:4d} | " + (f"{a['fano']:.2f} {a['ce']:.2f} [{a['ce_ci'][0]:.2f},{a['ce_ci'][1]:.2f}]" if a else " —                       ")
    line += f" | {f(s8)} | {f(u4)} | {f(c)} | " + (f"{dd['D_within']:.3f} {dd['D_between']:.3f} {dd['I_iso_per_dim']:.3f}" if dd else "—")
    print(line)
json.dump(res, open("out/b21.json", "w"), indent=1, default=float)
