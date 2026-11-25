"""Task 4: laws L1-L4 for the not-copied fraction; leave-one-configuration-out CV; AICc/BIC; bootstrap of the CV differences;
predictions with intervals (N=247,1000,2456 with K=32; and 128x128) on the 4-voice metric (grid unseen) and the 38-voice metric (wide).
Usage: python3 -I dados_04_leis_binario.py <results_dir> <out_dir> [B]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from dados_comum import load, main_cond
from dados_leis_bin import *
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 50)
df = load(sys.argv[1]); out = sys.argv[2]; B = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
d = main_cond(df); rng = np.random.default_rng(44)
seed = json.load(open(os.path.join(out, "seed_sd.json")))
SD_SEED_Q = seed["unseen|copied"]["sd_seed_total"]          # between-seed SD of the fraction (4 voices)
def cube(dd, col): return dd.pivot(index="reference", columns="sent", values=col).sort_index().values
def stack(runs, grp, col): return np.stack([cube(d[(d.run == r) & (d.group == grp)], col) for r in runs])
PRED = [(247, 32), (1000, 32), (2456, 32), (128, 128)]

def analyse(label, runs, grp, laws, ref_law, extra_noise, resample_spk):
    meta = d[d.run.isin(runs)].groupby("run")[["N", "K"]].first().loc[runs]
    N = meta.N.values.astype(float); K = meta.K.values.astype(float)
    keys = [f"{int(a)}x{int(b)}" for a, b in zip(N, K)]; cfg = np.array([sorted(set(keys)).index(x) for x in keys])
    Yc = stack(runs, grp, "copied"); Ym = stack(runs, grp, "m")
    q = 1 - Yc.mean((1, 2)); mu = Ym.mean((1, 2)); mix = mixture(Ym)
    ev = evaluate(laws, N, K, q, mu, mix, cfg)
    t = pd.DataFrame({nm: {k: v for k, v in e.items() if k != "f"} for nm, e in ev.items()}).T
    t["dAICc"] = t.aicc - t.aicc.min(); t["dBIC"] = t.bic - t.bic.min()
    for (n_, k_) in PRED: t[f"copia {n_}x{k_}"] = [1 - ev[nm]["f"](n_, k_)[0] for nm in t.index]
    print(f"\n=== {label}: {len(runs)} runs, {len(set(cfg))} configurations; probit mixture: u_s={np.round(mix[0],3)}, sig={mix[1]:.3f} ===")
    print(t.drop(columns=["aicc", "bic"]).astype(float).round(3).to_string())
    for nm in laws:
        f = ev[nm]["f"]
        if hasattr(f, "par"): print(f"   {nm}: c={f.par[0]:.3f} alpha={f.par[1]:.2f} beta={f.par[2]:.2f}")
    # bootstrap: (A) evaluation set (speakers x utterances, paired across runs); (B) A + seed noise per run
    a, b = Yc.shape[1], Yc.shape[2]
    res = {}
    for mode, sd_extra in [("aval", 0.0), ("aval+semente", extra_noise)]:
        cvs = np.empty((B, len(laws))); preds = np.empty((B, len(laws), len(PRED)))
        for i in range(B):
            si = rng.integers(0, a, a) if resample_spk else np.arange(a); ti = rng.integers(0, b, b)
            yc = Yc[:, si][:, :, ti]; ym = Ym[:, si][:, :, ti]
            noise = rng.normal(0, sd_extra, len(runs))
            qb = np.clip(1 - yc.mean((1, 2)) + noise, 0, 1)
            # equivalent seed noise on the margin (via the local slope d q / d mu ~ phi/sig)
            mub = ym.mean((1, 2)) - noise * mix[1] / np.maximum(norm.pdf(norm.ppf(np.clip(qb, .02, .98))), 1e-3) if sd_extra > 0 else ym.mean((1, 2))
            mx = mixture(ym)
            for j, (nm, (fit, k)) in enumerate(laws.items()):
                e = loco(fit, N, K, qb, mub, mx, cfg) - qb; cvs[i, j] = np.sqrt(np.mean(e ** 2))
                f = fit(N, K, qb, mub, mx); preds[i, j] = [1 - f(n_, k_)[0] for n_, k_ in PRED]
        names = list(laws); r = names.index(ref_law)
        print(f"\n -- bootstrap {mode} (resample speakers={resample_spk}; utterances always) (B={B}, extra noise per run sd={sd_extra:.3f}) --")
        rows = []
        for j, nm in enumerate(names):
            dd = cvs[:, j] - cvs[:, r]
            rows.append(dict(lei=nm, cv_med=np.median(cvs[:, j]), cv_lo=np.percentile(cvs[:, j], 2.5), cv_hi=np.percentile(cvs[:, j], 97.5),
                             dif_vs_ref=np.median(dd), dif_lo=np.percentile(dd, 2.5), dif_hi=np.percentile(dd, 97.5), P_pior=np.mean(dd > 0),
                             P_melhor_de_todas=np.mean(cvs.argmin(1) == j)))
        print(pd.DataFrame(rows).round(4).to_string()); print(f"    (reference for the differences: {ref_law})")
        pr = []
        for j, nm in enumerate(names):
            for p_, (n_, k_) in enumerate(PRED):
                x = preds[:, j, p_]
                pr.append(dict(lei=nm, cfg=f"{n_}x{k_}", ponto=1 - ev[nm]["f"](n_, k_)[0], lo=np.percentile(x, 2.5), hi=np.percentile(x, 97.5)))
        pr = pd.DataFrame(pr); res[mode] = pr
        print(pr.pivot(index="lei", columns="cfg", values="ponto").round(3).astype(str).add(" [") .add(pr.pivot(index="lei", columns="cfg", values="lo").round(2).astype(str)).add(", ").add(pr.pivot(index="lei", columns="cfg", values="hi").round(2).astype(str)).add("]").to_string())
    return t, res, (N, K, q, mu, mix, cfg)

plain = d[d.variant == "plain"]
runs = sorted(plain.run.unique(), key=lambda r: (plain[plain.run == r].N.iloc[0], plain[plain.run == r].K.iloc[0], r))
tU, rU, dataU = analyse("4 new voices (unseen), plain grid", runs, "unseen", LAWS_NK, "L2: a + A/sqrtN + C/sqrtK", SD_SEED_Q, False)
runs3 = [r for r in runs if not r.startswith("n2_")]
tU2, rU2, _ = analyse("4 new voices, without N=2", runs3, "unseen", LAWS_NK, "L2: a + A/sqrtN + C/sqrtK", SD_SEED_Q, False)
W = ["n8_k32_s1-wide", "n32_k32_s1-wide", "n128_k32_s1-wide"]
# for 38 voices the seed SD of the fraction was not measured; use the main seed effect estimated on the 4 voices (sd_seed_main)
sd38 = seed["unseen|copied"]["sd_seed_main"]
tW, rW, dataW = analyse("38 external voices (wide), K=32, 3 runs, 1 seed", W, "external", LAWS_N, "L2n: a + A/sqrtN", sd38, True)
# parametric identifiability (4 voices): generate from each fitted law + seed noise, choose by CV and by AICc
N, K, q, mu, mix, cfg = dataU
print(f"\n=== identifiability, 4 voices: y = fitted law + N(0, {SD_SEED_Q:.3f}^2) per run; B=200 ===")
names = list(LAWS_NK); fits = {nm: LAWS_NK[nm][0](N, K, q, mu, mix) for nm in names}
conf_cv = np.zeros((len(names), len(names))); conf_aic = np.zeros_like(conf_cv)
slope = norm.pdf(norm.ppf(np.clip(q, .02, .98))) / mix[1]
for i, nm in enumerate(names):
    yh = np.clip(fits[nm](N, K), 0, 1)
    for _ in range(200):
        e = rng.normal(0, SD_SEED_Q, len(q)); qs = np.clip(yh + e, 0, 1); ms = mu - e / np.maximum(slope, 1)
        ev = evaluate(LAWS_NK, N, K, qs, ms, mix, cfg)
        conf_cv[i, np.argmin([ev[x]["cv"] for x in names])] += 1; conf_aic[i, np.argmin([ev[x]["aicc"] for x in names])] += 1
sh = ["L1", "L2", "L3", "L4", "L4m", "L4m'"]
print("choice by CV (rows: generating law)"); print(pd.DataFrame(conf_cv / 200, index=sh, columns=sh).round(2).to_string())
print("choice by AICc"); print(pd.DataFrame(conf_aic / 200, index=sh, columns=sh).round(2).to_string())
pd.concat({"unseen": tU.drop(columns=[]), "unseen_semN2": tU2, "wide38": tW}).to_csv(os.path.join(out, "leis_binario.csv"))
pd.concat({f"{a}|{m}": r_ for a, rr in [("unseen", rU), ("wide38", rW)] for m, r_ in rr.items()}).to_csv(os.path.join(out, "previsoes_binario.csv"))
