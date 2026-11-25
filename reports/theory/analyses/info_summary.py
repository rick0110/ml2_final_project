"""Builds summary tables from out/*.json, fits rate vs (log N, log K), and the cross-run scaling of the
(noise-corrected) distance from the 38 external references to the nearest trained speaker.
usage: python3 -I info_summary.py OUT"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from info_common import md_table
out = sys.argv[1]
rate = json.load(open(os.path.join(out, "rate.json"))); sp = json.load(open(os.path.join(out, "speaker.json")))
sp2 = json.load(open(os.path.join(out, "speaker2.json"))); at = json.load(open(os.path.join(out, "attractor.json")))
rng = np.random.default_rng(3); txt = ""
# ---- rate vs N, K (base runs only)
base = [r for r in rate.values() if r["variant"] == ""]
for key in ("rate_total", "rate_lib", "rate_lj", "I_gauss_full"):
    y = np.array([r[key] for r in base]); X = np.column_stack([np.ones(len(base)), np.log2([r["N"] for r in base]), np.log2([r["K"] for r in base])])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); resid = y - X @ b
    s2 = resid @ resid / (len(y) - 3); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    txt += f"{key}: {b[0]:.2f} + {b[1]:.2f}(±{se[1]:.2f})·log2N + {b[2]:.2f}(±{se[2]:.2f})·log2K, rmse {np.sqrt(s2):.2f}\n"
# per-dim profile
pdm = np.array([r["per_dim"] for r in base]); srt = np.sort(pdm, 1)[:, ::-1]
txt += "per-dim rate sorted (mean over base runs): " + " ".join(f"{v:.2f}" for v in srt.mean(0)) + "\n"
txt += "PR of per-dim rate: " + ", ".join(f"{r['run']}:{r['pr_rate']:.1f}" for r in base) + "\n"
# ---- speaker info table
rows = []
for run, r in sp.items():
    if not r.get("enough") or r["variant"] not in ("", "wide"): continue
    q = sp2.get(run, {})
    rows.append([run, r["S"], f"{r['n']/r['S']:.1f}", f"{r['log2S']:.2f}", f"{r['H']:.2f}",
                 f"{r['I_plug']:.1f}", f"{q.get('I_bc', np.nan):.1f} ± {q.get('I_bc_sd', np.nan):.1f}", f"{q.get('I_bc_chan', np.nan):.1f}",
                 f"{r['I_plug_perm']:.2f} / {r['I_vc_perm']:.2f}",
                 f"{r['lda_acc']:.2f}", f"{r['I_ce']:.2f}", f"{r['I_fano']:.2f}", f"{r.get('I_ce_lr', np.nan):.2f}", f"{r['I_ce_perm']:.2f}",
                 f"{r['I_bin_ce']:.2f}/{r['Hbin']:.2f}", f"{r['d_lj_lib']:.1f}"])
hdr = ["run", "S", "utts/spk.", "log2 S", "H(S)", "Gauss plug-in", "Gauss corr. (mu)", "Gauss corr. (z)", "perm. plug/vc", "acc LDA-LOO",
       "I_CE (LDA)", "I_Fano", "I_CE (logreg 5-fold)", "I_CE perm.", "LJ×Libri I/H", "dist. LJ–Libri (sd)"]
tab_sp = md_table(hdr, rows)
# ---- directions
rows = []
for run, q in sp2.items():
    r = sp[run]
    if q["S"] < 8: continue
    rows.append([run, q["S"], f"{q['pr_raw']:.1f}", f"{q.get('pr_Sbw', np.nan):.1f}", f"{r['pr_snr']:.1f}", r["n_snr1"],
                 sum(1 for v in q["snr_chan"] if v > 1), q["k50_chan"], q["k90_chan"], q["k99_chan"], q["k90_mu"],
                 f"{r['alpha_raw']:.2f} [{r['alpha_raw_ci'][0]:.2f}, {r['alpha_raw_ci'][1]:.2f}]" if r.get("alpha_raw_ci") else f"{r['alpha_raw']:.2f}",
                 f"{q['gain_R4_mu']:.1f} / {q['gain_R4_chan']:.1f}"])
hdr = ["run", "S", "PR(Σb)", "PR(W^-1/2 Σb W^-1/2)", "PR(SNR)", "dir. SNR>1 (mu)", "dir. SNR>1 (z)", "k 50% bits (z)", "k 90% (z)", "k 99% (z)", "k 90% (mu)", "α power law (Σb, 16 dir.)", "predicted R=4 gain (mu/z, bits)"]
tab_dir = md_table(hdr, rows)
# ---- ID
rows = []
for run, q in sp2.items():
    if "twonn_denoised" not in q: continue
    r = sp[run]
    rows.append([run, q["S"], f"{r['twonn_raw']:.1f}", f"{r['twonn_noise_only']:.1f}", f"{q['twonn_denoised']:.1f} [{q['twonn_denoised_ci'][0]:.1f}, {q['twonn_denoised_ci'][1]:.1f}]",
                 f"{q['twonn_null']:.1f} [{q['twonn_null_ci'][0]:.1f}, {q['twonn_null_ci'][1]:.1f}]",
                 f"{q['lb_denoised']:.1f} [{q['lb_denoised_ci'][0]:.1f}, {q['lb_denoised_ci'][1]:.1f}]", f"{q['lb_null']:.1f}",
                 (f"{q['cover_slope']:.3f} [{q['cover_slope_ci'][0]:.3f}, {q['cover_slope_ci'][1]:.3f}] → d≈{1/q['cover_slope']:.1f}" if "cover_slope" in q else "—"),
                 (f"{q['cover_slope_null']:.3f} ± {q['cover_slope_null_sd']:.3f}" if "cover_slope" in q else "—")])
hdr = ["run", "S", "TwoNN raw means", "TwoNN noise only", "TwoNN denoised means [80% CI subsample]", "TwoNN Gaussian null [95% CI]", "Levina–Bickel k=5 denoised", "LB null", "coverage exponent γ (r_NN ∝ m^-γ)", "γ Gaussian null"]
tab_id = md_table(hdr, rows)
# ---- cross-run NN distance of external references, noise-corrected
rows = []; dd = {}
for run in ["n8_k32_s1-wide", "n32_k32_s1-wide", "n128_k32_s1-wide"]:
    ps = at[run]["per_speaker"]; dnn = np.array([p["d_nn_w"] for p in ps]); dlj = np.array([p["d_lj_w"] for p in ps])
    S = sp[run]["S"]; nps = sp[run]["n"] / S
    floor = 32 * (1 + 1 / nps)          # E||e||^2 of a single utterance + mean noise, whitened units
    corr = np.sqrt(np.clip(dnn**2 - floor, 0, None))
    b = [np.mean(np.sqrt(np.clip(dnn[i]**2 - floor, 0, None))) for i in [rng.integers(0, len(dnn), len(dnn)) for _ in range(2000)]]
    dd[run] = (dnn, corr)
    rows.append([run, f"{dnn.mean():.2f}", f"{np.sqrt(floor):.2f}", f"{corr.mean():.2f} [{np.percentile(b,2.5):.2f}, {np.percentile(b,97.5):.2f}]",
                 f"{dlj.mean():.1f}", f"{at[run]['lj_center_to_lib_center_w']:.1f}", f"{at[run]['copy_mean']:.2f}"])
tab_nn = md_table(["run", "d_NN (whitened)", "noise floor √E‖e‖²", "corrected d_NN [95% CI]", "d_LJ", "dist. LJ centre–Libri centre", "mean copy"], rows)
N = np.array([8, 32, 128]); m = np.array([dd[r][1].mean() for r in dd]); mr = np.array([dd[r][0].mean() for r in dd])
txt += f"cross-run slope of corrected d_NN vs N: {-np.polyfit(np.log(N), np.log(m), 1)[0]:.3f}; raw: {-np.polyfit(np.log(N), np.log(mr), 1)[0]:.3f}\n"
# ---- attractor table
rows = []
for run in ["n8_k32_s1-wide", "n32_k32_s1-wide", "n128_k32_s1-wide"]:
    a = at[run]; rho, ci, lb = a["rho"], a["ci"], a["loo_bits"]
    rows.append([run, f"{a['copy_mean']:.2f}", f"{rho['copy~d_lj_w']:.2f} [{ci['copy~d_lj_w'][0]:.2f}, {ci['copy~d_lj_w'][1]:.2f}]",
                 f"{rho['copy~d_nn_w']:.2f} [{ci['copy~d_nn_w'][0]:.2f}, {ci['copy~d_nn_w'][1]:.2f}]",
                 f"{rho['copy~logf0']:.2f} [{ci['copy~logf0'][0]:.2f}, {ci['copy~logf0'][1]:.2f}]",
                 f"{ci['P(|rho_lj|>|rho_nn|)']:.2f}", f"{rho['copy~d_lj_w|f0']:.2f}", f"{rho['copy~d_nn_w|d_lj_w']:.2f}",
                 f"{rho['sim_ref~d_lj_w']:.2f} [{ci['sim_ref~d_lj_w'][0]:.2f}, {ci['sim_ref~d_lj_w'][1]:.2f}]",
                 f"{rho['sim_def~d_lj_w']:.2f} [{ci['sim_def~d_lj_w'][0]:.2f}, {ci['sim_def~d_lj_w'][1]:.2f}]"])
tab_at = md_table(["run", "copy", "ρ(copy, d_LJ)", "ρ(copy, d_NN)", "ρ(copy, log F0)", "P(abs ρ_LJ > abs ρ_NN)", "partial ρ d_LJ given F0", "partial ρ d_NN given d_LJ", "ρ(sim_ref, d_LJ)", "ρ(sim_default, d_LJ)"], rows)
rows = []
for run in ["n8_k32_s1-wide", "n32_k32_s1-wide", "n128_k32_s1-wide"]:
    a = at[run]; lb = a["loo_bits"]
    rows.append([run] + [f"{lb[k]:.3f}" for k in ["intercepto", "d_lj", "d_nn", "f0", "d_lj+d_nn", "d_lj+f0", "d_lj+d_nn+f0"]] +
                [f"{a['d50_w']:.1f} [{a['d50_ci'][0]:.1f}, {a['d50_ci'][1]:.1f}]", f"{a['width_w']:.1f} [{a['width_ci'][0]:.1f}, {a['width_ci'][1]:.1f}]",
                 f"{a['overdispersion']:.0f}", f"{a['frac_all_or_none']:.2f}"])
p = at["pooled"]
rows.append(["pooled (run fixed effect, LOSO)", f"{p['run']['loso_bits']:.3f}", f"{p['d_lj']['loso_bits']:.3f}", f"{p['d_nn']['loso_bits']:.3f}", f"{p['f0']['loso_bits']:.3f}",
             f"{p['d_lj+d_nn']['loso_bits']:.3f}", "—", f"{p['d_lj+d_nn+f0']['loso_bits']:.3f}", "—", "—", "—", "—"])
tab_lb = md_table(["run", "intercept only", "d_LJ", "d_NN", "log F0", "d_LJ+d_NN", "d_LJ+F0", "d_LJ+d_NN+F0", "d50 (whitened)", "width 1/β", "overdispersion", "all-or-none fraction"], rows)
txt += "pooled coefs (z-scored, d_lj,d_nn,f0): " + json.dumps(p["d_lj+d_nn+f0"]["coef"]) + " CI " + json.dumps(p["coef_ci_d_lj+d_nn+f0"]) + "\n"
txt += "pooled within-run spearman: " + json.dumps(p["spearman_within_run"]) + "\n"
open(os.path.join(out, "summary_tables.md"), "w").write("\n## speaker\n" + tab_sp + "\n## dirs\n" + tab_dir + "\n## id\n" + tab_id + "\n## nn\n" + tab_nn + "\n## attractor\n" + tab_at + "\n## loo bits\n" + tab_lb + "\n## misc\n```\n" + txt + "```\n")
print(txt)
