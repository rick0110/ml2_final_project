"""b24: C(N,K) de fidelidade no conjunto comum (12 vozes: 8 treinadas + 4 novas) e no de 16 (com 4 externas),
inclinacoes por duplicacao com bootstrap pareado de vozes, decomposicao I = H(Y) - H(Y|S), q x C, forca 0.6 e R.
Uso: python3 -I b24_lei.py out/b0.npz out/b12_emb.npz out/b20_outs.npz out/b20_refs.npz <results_root> [B]"""
import sys, os, re, csv, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr
from bt_lib import load
S = load(sys.argv[1]); RES = sys.argv[5]; B = int(sys.argv[6]) if len(sys.argv) > 6 else 1000; rng = np.random.default_rng(21)
def ld(p):
    d = np.load(p, allow_pickle=False); E = d["emb"] / np.linalg.norm(d["emb"], axis=1, keepdims=True); return list(d["keys"].astype(str)), E
OUT, cent = {}, {}
for f in sys.argv[2:5]:
    ks, E = ld(f)
    for i, k in enumerate(ks):
        p = k.split("|")
        if p[0] == "out": ref, sent = p[3].rsplit("__", 1); OUT.setdefault((p[1], p[2]), {}).setdefault(ref, {})[sent] = E[i]
        elif p[0] == "oth": cent.setdefault(p[1], []).append(E[i])
        elif p[0] == "ref": cent.setdefault("REF|" + p[2], []).append(E[i])
CEN = {s: np.mean(v, 0) / np.linalg.norm(np.mean(v, 0)) for s, v in cent.items() if not s.startswith("REF|")}
V12 = sorted(x["ref"] for x in S["n8_k128_s1-spk"]["refs"] if x["grp"] in ("seen", "unseen"))
SEEN = [v for v in V12 if v in [x["ref"] for x in S["n8_k128_s1-spk"]["refs"] if x["grp"] == "seen"]]; UNS = [v for v in V12 if v not in SEEN]
EXT4 = sorted(x["ref"] for x in S["n128_k32_s1"]["refs"] if x["grp"] == "external"); V16 = V12 + EXT4
def meta(run):
    m = re.match(r"n(\d+)_k(\d+)_s(\d+)(?:-(.*))?$", run); return int(m.group(1)), int(m.group(2)), m.group(4) or "base"
def fano(acc, n):
    pe = min(max(1 - acc, 1e-12), 1 - 1e-12); return np.log2(n) - (-(pe*np.log2(pe)+(1-pe)*np.log2(1-pe))) - pe*np.log2(n-1)
BETAS = np.linspace(1, 150, 150)
def rows_of(run, cond, vs):
    o = OUT.get((run, cond), {})
    if not all(v in o for v in vs) or not all(v.split("_")[0] in CEN for v in vs): return None
    C = np.array([CEN[v.split("_")[0]] for v in vs]); Y, t = [], []
    for j, v in enumerate(vs):
        for e in o[v].values(): Y.append(e); t.append(j)
    Y, t = np.array(Y), np.array(t); sim = Y @ C.T
    tab = np.array([[-(b * sim[i, t[i]] - logsumexp(b * sim[i])) / np.log(2) for i in range(len(t))] for b in BETAS])
    return dict(t=t, hit=(sim.argmax(1) == t).astype(float), tab=tab, n=len(vs))
def summ(R, pick=None):
    """CE-LOVO e Fano para um conjunto de vozes (pick = indices de vozes, com repeticao, para bootstrap)"""
    n = R["n"]; ii = np.arange(len(R["t"])) if pick is None else np.concatenate([np.where(R["t"] == p)[0] for p in pick])
    tab = R["tab"][:, ii]; tot = tab.sum(1); ce = np.mean([tab[np.argmin(tot - tab[:, j]), j] for j in range(tab.shape[1])])
    return np.log2(n) - ce, fano(R["hit"][ii].mean(), n)
runs = sorted(set(r for r, c in OUT), key=lambda r: (meta(r)[0], meta(r)[1], r))
def qobs(run, grps, cond="strength_1.0"):
    p = os.path.join(RES, run, "step_2500", cond, "results.csv")
    if not os.path.exists(p): return None
    k = n = 0
    with open(p, encoding="utf-8") as f:
        for x in csv.DictReader(f):
            if x["reference_emotion"] in grps: n += 1; k += float(x["sim_ref"]) > float(x["sim_default"])
    return 1 - k / n if n else None
print(f"centroides reais: V12 {sum(v.split('_')[0] in CEN for v in V12)}/12, EXT4 {sum(v.split('_')[0] in CEN for v in EXT4)}/4")
print("\n(1) fidelidade I(saida; voz) por identificacao entre centroides de fala real; forca 1.0, R=1. [IC 95% bootstrap de vozes]")
print(f"{'treino':22s} {'N':>4s} {'K':>4s} {'var':8s} | V12 (cap 3.58): EC [IC]   Fano | 8 treinadas (cap 3): EC | 4 novas (cap 2): EC | q(trein.+novas)")
TAB = {}
for r in runs:
    N, K, var = meta(r)
    R12 = rows_of(r, "strength_1.0", V12); Rs = rows_of(r, "strength_1.0", SEEN); Ru = rows_of(r, "strength_1.0", UNS)
    if R12 is None and Ru is None: continue
    line = f"{r:22s} {N:4d} {K:4d} {var:8s} | "
    if R12:
        ce, fa = summ(R12); bs = np.array([summ(R12, rng.integers(0, 12, 12))[0] for _ in range(B // 5)])
        line += f"{ce:.2f} [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}] {fa:.2f}"
        TAB[r] = dict(N=N, K=K, var=var, ce=ce, fano=fa, ci=np.percentile(bs, [2.5, 97.5]).tolist())
    else: line += "      —                  "
    cs = summ(Rs)[0] if Rs else np.nan; cu = summ(Ru)[0] if Ru else np.nan
    q = qobs(r, ("seen", "unseen")); qu = qobs(r, ("unseen",)); qs = qobs(r, ("seen",))
    if r in TAB: TAB[r].update(ce_seen=cs, ce_uns=cu, q=q, q_uns=qu, q_seen=qs)
    else: TAB[r] = dict(N=N, K=K, var=var, ce_uns=cu, q_uns=qu)
    print(line + f" | {cs:.2f} | {cu:.2f} | {q if q is None else round(q,3)}")
# slopes: paired bootstrap over the 12 voices
def fitslope(sel, key="ce", vs=None):
    rr = [r for r in sel]; X = np.array([[1, np.log2(TAB[r]["N"]), np.log2(TAB[r]["K"])] for r in rr]); y = np.array([TAB[r][key] for r in rr])
    b = np.linalg.lstsq(X, y, rcond=None)[0]; return b, X, y
sel = [r for r in TAB if "ce" in TAB[r] and TAB[r]["N"] >= 8 and not r.endswith("-wide")]
b, X, y = fitslope(sel)
Rr = {r: rows_of(r, "strength_1.0", V12) for r in sel}
bs = []
for _ in range(B // 5):
    pick = rng.integers(0, 12, 12); yy = np.array([summ(Rr[r], pick)[0] for r in sel]); bs.append(np.linalg.lstsq(X, yy, rcond=None)[0])
bs = np.array(bs); resid = y - X @ b
print(f"\n(2) lei I = c + s_N log2 N + s_K log2 K (V12, EC; {len(sel)} treinos com N>=8, base+variantes; -wide excluido por ter 1 frase):")
print(f"    c = {b[0]:.2f}; s_N = {b[1]:+.3f} bit/duplicacao [IC {np.percentile(bs[:,1],2.5):+.3f}, {np.percentile(bs[:,1],97.5):+.3f}]; s_K = {b[2]:+.3f} [IC {np.percentile(bs[:,2],2.5):+.3f}, {np.percentile(bs[:,2],97.5):+.3f}]; RMSE resid {np.sqrt(np.mean(resid**2)):.2f} bits")
selb = [r for r in sel if TAB[r]["var"] == "base"]
print("    so treinos base: " + ", ".join(f"{r} {TAB[r]['ce']:.2f}" for r in selb))
# seen and unseen separately
for key in ["ce_seen", "ce_uns"]:
    s2 = [r for r in TAB if key in TAB[r] and np.isfinite(TAB[r][key]) and TAB[r]["N"] >= 8 and not r.endswith("-wide")]
    Xs = np.array([[1, np.log2(TAB[r]["N"]), np.log2(TAB[r]["K"])] for r in s2]); ys = np.array([TAB[r][key] for r in s2])
    bb = np.linalg.lstsq(Xs, ys, rcond=None)[0]; print(f"    {key}: s_N = {bb[1]:+.3f}, s_K = {bb[2]:+.3f} bit/duplicacao ({len(s2)} treinos)")
# q vs C
qq = [(TAB[r]["ce"], TAB[r]["q"], TAB[r]["var"], r) for r in TAB if "ce" in TAB[r] and TAB[r].get("q") is not None]
print(f"\n(3) q x C (V12): Spearman = {spearmanr([a for a,_,_,_ in qq],[b_ for _,b_,_,_ in qq]).correlation:.2f} em {len(qq)} treinos")
for a, b_, v, r in sorted(qq): print(f"    {r:22s} C={a:.2f}  q={b_:.3f}")
# 4 novas incl. N=2
print("\n    4 novas (cap 2 bits), inclui N=2: " + "; ".join(f"{r}: C {TAB[r]['ce_uns']:.2f} q {TAB[r]['q_uns']:.2f}" for r in TAB if np.isfinite(TAB[r].get('ce_uns', np.nan)) and TAB[r].get('q_uns') is not None))
# (4) strength 0.6 and R on V16 / V12
print("\n(4) forca 0.6 e R: fidelidade EC (V16 = 12 + 4 externas, cap 4; V12 nos treinos variantes)")
for r in runs:
    line = f"  {r:22s}"
    for cond in ["strength_1.0", "strength_0.6", "strength_1.0_r4", "strength_1.0_r16"]:
        vs = V16 if all(v in OUT.get((r, cond), {}) for v in V16) else V12
        R_ = rows_of(r, cond, vs)
        if R_: line += f" | {cond[9:] or '1.0'}: {summ(R_)[0]:.2f}/{np.log2(R_['n']):.0f}"
    print(line)
# (5) decomposition, Gaussian in a common PCA basis
allY = np.array([e for (r, c), o in OUT.items() for v in V12 if v in o for e in o[v].values()])
mu0 = allY.mean(0); U, s_, Vt = np.linalg.svd(allY - mu0, full_matrices=False)
print("\n(5) decomposicao I = H(Y) - H(Y|S) (gaussiana, base PCA comum k dims; nats->bits); s0/s1 dao H(Y|S)")
for k in [3, 6]:
    P = Vt[:k]; print(f"  k = {k}")
    rows = []
    for r in runs:
        o = OUT.get((r, "strength_1.0"), {}); vs = [v for v in V12 if v in o and "s1" in o[v]]
        if len(vs) < 8: continue
        A = np.array([o[v]["s0"] for v in vs]) - mu0; Bm = np.array([o[v]["s1"] for v in vs]) - mu0; A, Bm = A @ P.T, Bm @ P.T
        Sw = (A - Bm).T @ (A - Bm) / (2 * len(vs)); Y = np.vstack([A, Bm]); St = np.cov(Y.T)
        Ht = 0.5 * np.log2(np.linalg.det(2 * np.pi * np.e * St)); Hw = 0.5 * np.log2(np.linalg.det(2 * np.pi * np.e * Sw))
        N, K, var = meta(r); rows.append((N, K, Ht, Hw)); print(f"    {r:22s} N={N:3d} K={K:3d}: H(Y) {Ht:6.2f}  H(Y|S) {Hw:6.2f}  I_G {Ht-Hw:5.2f} bits")
    rows = np.array([x for x in rows if x[0] >= 8]); Xd = np.column_stack([np.ones(len(rows)), np.log2(rows[:, 0]), np.log2(rows[:, 1])])
    for j, nm in [(2, "H(Y)"), (3, "H(Y|S)")]:
        bb = np.linalg.lstsq(Xd, rows[:, j], rcond=None)[0]; print(f"    {nm}: inclinacao {bb[1]:+.3f} bit por duplicacao de N, {bb[2]:+.3f} por duplicacao de K")
json.dump(TAB, open("out/b24.json", "w"), indent=1, default=float)
