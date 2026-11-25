"""b13: voice capacity of the decoder measured in the audio: I(output voice; reference identity) via identification.
Usage: python3 -I b13_capacidade.py out/b12_emb.npz <results_root> [B]"""
import sys, os, csv, json
import numpy as np
from scipy.special import logsumexp
d = np.load(sys.argv[1], allow_pickle=False); RES = sys.argv[2]; B = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
keys = d["keys"].astype(str); E = d["emb"]; E = E / np.linalg.norm(E, axis=1, keepdims=True)
rng = np.random.default_rng(1)
K = [k.split("|") for k in keys]
ref = {k[2]: E[i] for i, k in enumerate(K) if k[0] == "ref"}           # ref utt id -> emb
oth = {}
for i, k in enumerate(K):
    if k[0] == "oth": oth.setdefault(k[1], []).append(E[i])
spk_of = {r: r.split("_")[0] for r in ref}
EXT = sorted(ref)                                                         # 38 externals with reference audio
S = len(EXT); H = np.log2(S)
cent = np.array([np.mean(oth[spk_of[r]], 0) for r in EXT]); cent /= np.linalg.norm(cent, axis=1, keepdims=True)
R_utt = np.array([ref[r] for r in EXT])
def outs(run, cond, sent="s0"):
    m = {}
    for i, k in enumerate(K):
        if k[0] == "out" and k[1] == run and k[2] == cond and k[3].endswith("__" + sent): m[k[3].rsplit("__", 1)[0]] = E[i]
    return m
def fano(acc, S):
    pe = min(max(1 - acc, 1e-12), 1 - 1e-12); return np.log2(S) - (-(pe*np.log2(pe)+(1-pe)*np.log2(1-pe))) - pe*np.log2(S-1)
def ce_bits(sim, beta):     # sim (n,S), true class on the diagonal
    L = beta * sim; return -np.mean(np.diag(L) - logsumexp(L, axis=1)) / np.log(2)
def row_ce(sim, i, beta):
    L = beta * sim[i]; return -(L[i] - logsumexp(L)) / np.log(2)
def lovo_rows(sim):         # CE per row, with beta chosen on the OTHER rows (no leakage)
    betas = np.linspace(1, 150, 150); n = len(sim)
    tab = np.array([[row_ce(sim, j, b) for j in range(n)] for b in betas])   # (nb, n)
    tot = tab.sum(1); out = np.empty(n)
    for i in range(n):
        k = np.argmin(tot - tab[:, i]); out[i] = tab[k, i]
    return out
def lovo_ce(sim): return float(lovo_rows(sim).mean())
def plugin_mm(pred, true, S):   # plug-in information of the confusion matrix with Miller-Madow correction
    n = len(pred); J = np.zeros((S, S)); np.add.at(J, (true, pred), 1); P = J / n
    px, py = P.sum(1), P.sum(0); nz = P > 0
    I = np.sum(P[nz] * np.log2(P[nz] / np.outer(px, py)[nz]))
    kxy, kx, ky = nz.sum(), (px > 0).sum(), (py > 0).sum()
    return I, I - (kxy - kx - ky + 1) / (2 * n * np.log(2))
def analyse(sim, label):
    true = np.arange(len(sim)); pred = sim.argmax(1); acc = float(np.mean(pred == true))
    rank = np.array([(sim[i] > sim[i, i]).sum() + 1 for i in range(len(sim))])
    rows = lovo_rows(sim); ce = rows.mean(); Ice = H - ce; If = fano(acc, S); Ip, Imm = plugin_mm(pred, true, S)
    bs = []
    for _ in range(B):
        ii = rng.integers(0, len(sim), len(sim)); a = np.mean(pred[ii] == true[ii])
        bs.append([fano(a, S), H - rows[ii].mean()])
    bs = np.array(bs)
    r = dict(label=label, acc=acc, top5=float(np.mean(rank <= 5)), mean_log2rank=float(np.mean(np.log2(rank))), I_fano=float(If), I_ce=float(Ice),
             I_plugin=float(Ip), I_plugin_MM=float(Imm), fano_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), ce_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist())
    print(f"  {label:42s} acc {acc:.3f} top5 {r['top5']:.2f} | Fano {If:5.2f} [{r['fano_ci'][0]:.2f},{r['fano_ci'][1]:.2f}] | CE(LOVO) {Ice:5.2f} [{r['ce_ci'][0]:.2f},{r['ce_ci'][1]:.2f}] | plug-in {Ip:.2f} (MM {Imm:.2f}) bits  / log2 S = {H:.2f}")
    return r
BETA = 30.0
res = {}
print(f"S = {S} external voices with audio; H = {H:.2f} bits")
print("\n(0) validation: cos(output s0, ref) computed here vs sim_ref from results.csv")
for run in ["n8_k32_s1-wide", "n128_k32_s1-wide"]:
    o = outs(run, "strength_1.0"); a, b = [], []
    with open(os.path.join(RES, run, "step_2500", "strength_1.0", "results.csv"), encoding="utf-8") as f:
        for x in csv.DictReader(f):
            if x["sample"].endswith("__s0") and x["reference"] in o and x["reference"] in ref:
                a.append(float(o[x["reference"]] @ ref[x["reference"]])); b.append(float(x["sim_ref"]))
    print(f"  {run}: n={len(a)} corr {np.corrcoef(a, b)[0,1]:.4f} max|diff| {np.max(np.abs(np.array(a)-np.array(b))):.4f}")
print("\n(1) verifier ceiling: held-out real utterance vs centroids (other utterances) / vs reference utterance")
oth_one = np.array([oth[spk_of[r]][0] for r in EXT])
# centroid without that utterance
cent_m = np.array([np.mean(oth[spk_of[r]][1:], 0) for r in EXT]); cent_m /= np.linalg.norm(cent_m, axis=1, keepdims=True)
BETA = 30.0
res["teto_frase_vs_centroide"] = analyse(oth_one @ cent_m.T, "real (held-out utterance) x centroid")
res["teto_frase_vs_ref"] = analyse(oth_one @ R_utt.T, "real (held-out utterance) x reference utterance")
print("\n(2) C(N): output (utterance s0) identified among the 38 voices")
for run in ["n8_k32_s1-wide", "n128_k32_s1-wide"]:
    for cond in ["strength_1.0", "strength_1.0_r4"]:
        o = outs(run, cond); O = np.array([o[r] for r in EXT])
        res[f"{run}|{cond}|cent"] = analyse(O @ cent.T, f"{run} {cond[9:] or '1.0'} x centroid")
        res[f"{run}|{cond}|ref"] = analyse(O @ R_utt.T, f"{run} {cond[9:] or '1.0'} x ref-utterance")
# paired bootstrap of difference N=128 - N=8 (centroid, R1) and R4 - R1
def acc_ce(run, cond):
    o = outs(run, cond); O = np.array([o[r] for r in EXT]); sim = O @ cent.T
    return (sim.argmax(1) == np.arange(S)).astype(float), lovo_rows(sim)
a8, c8 = acc_ce("n8_k32_s1-wide", "strength_1.0"); a128, c128 = acc_ce("n128_k32_s1-wide", "strength_1.0")
a8r, c8r = acc_ce("n8_k32_s1-wide", "strength_1.0_r4"); a128r, c128r = acc_ce("n128_k32_s1-wide", "strength_1.0_r4")
for nm, x, y in [("C(128)-C(8), R=1", c8, c128), ("C(R4)-C(R1), N=8", c8, c8r), ("C(R4)-C(R1), N=128", c128, c128r)]:
    bs = [np.mean(x[ii] - y[ii]) for ii in (rng.integers(0, S, S) for _ in range(B))]
    print(f"  {nm}: diff CE-bound = {np.mean(x-y):+.2f} bits, 95% CI [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]")
    res["diff|" + nm] = [float(np.mean(x - y)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
print("\n(3) output-output (no reference): utterance s0 of run A identified among the s0 utterances of run B / condition B")
pairs = [("n8_k32_s1-wide", "strength_1.0", "n8_k32_s1-wide", "strength_1.0_r4"), ("n128_k32_s1-wide", "strength_1.0", "n128_k32_s1-wide", "strength_1.0_r4"),
         ("n8_k32_s1-wide", "strength_1.0", "n128_k32_s1-wide", "strength_1.0")]
for ra, ca, rb, cb in pairs:
    oa, ob = outs(ra, ca), outs(rb, cb); ids = sorted(set(oa) & set(ob))
    A = np.array([oa[i] for i in ids]); Bm = np.array([ob[i] for i in ids]); S_ = len(ids)
    acc = np.mean((A @ Bm.T).argmax(1) == np.arange(S_)); print(f"  {ra}:{ca[9:] or '1.0'} x {rb}:{cb[9:] or '1.0'}  ({S_} voices, includes trained/new): acc {acc:.3f}  Fano {fano(acc, S_):.2f} / log2 {np.log2(S_):.2f}")
print("\n(4) plain runs (16 voices: 8 trained, 4 new, 4 external): utterance s0 x utterance s1 of the SAME condition (output-output identification)")
for run in ["n128_k8_s1", "n128_k32_s1", "n200_k32_s1"]:
    for cond in ["strength_1.0", "strength_0.6", "strength_1.0_r4", "strength_1.0_r16"]:
        o0, o1 = outs(run, cond, "s0"), outs(run, cond, "s1"); ids = sorted(set(o0) & set(o1))
        if not ids: continue
        A = np.array([o0[i] for i in ids]); Bm = np.array([o1[i] for i in ids]); S_ = len(ids); sim = A @ Bm.T
        acc = np.mean(sim.argmax(1) == np.arange(S_)); ce = lovo_ce(sim)
        offd = sim[~np.eye(S_, dtype=bool)].mean(); diag = np.diag(sim).mean()
        res[f"plain|{run}|{cond}"] = dict(acc=float(acc), fano=float(fano(acc, S_)), ce=float(np.log2(S_) - ce), diag=float(diag), offd=float(offd))
        print(f"  {run:12s} {cond:16s}: acc {acc:.3f} Fano {fano(acc,S_):.2f} CE {np.log2(S_)-ce:.2f} / {np.log2(S_):.2f} bits | cos same {diag:.3f} others {offd:.3f}")
json.dump(res, open("out/b13.json", "w"), indent=1, default=float)
