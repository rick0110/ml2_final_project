"""All site figures, using the validated reference palette (light surface)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).parent
ANA = HERE / "analysis"
IMG = Path("/home/al.richard.viana/Documentos/ml2_final_project/docs/assets/img")
IMG.mkdir(parents=True, exist_ok=True)

# ---- palette (reference instance, light mode) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AX = "#c3c2b7"
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
VIOLET, RED, MAGENTA, ORANGE = "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"

EMO_COLORS = {
    "neutro": MUTED, "alegria": YELLOW, "tristeza": BLUE, "raiva": RED,
    "medo": VIOLET, "surpresa": AQUA, "desgosto": GREEN,
}
EMO_ORDER = ["neutro", "alegria", "tristeza", "raiva", "medo", "surpresa", "desgosto"]

MEL_CMAP = LinearSegmentedColormap.from_list(
    "melblue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": BASELINE_AX,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titlecolor": INK, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 9,
    "axes.titlepad": 10, "figure.dpi": 160,
})


def save(fig, name):
    fig.savefig(IMG / name, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("saved", name)


def legend(ax, **kw):
    leg = ax.legend(frameon=False, **kw)
    for t in leg.get_texts():
        t.set_color(INK2)
    return leg


summary = {}

# ================================================== 1. LJ reconstructed curves
ck = json.load(open(HERE / "ckpt_eval.json"))
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
for ax, key, title in [(axes[0], "recon", "Perda de reconstrução (mel + gate)"),
                       (axes[1], "kl", "Divergência KL (nats/dim)")]:
    for name, color, label in [("baseline_lj", AQUA, "Baseline (última saída da GRU)"),
                               ("attn_lj", BLUE, "Com atenção temporal")]:
        pts = [(p["iteration"], p[key]) for p in ck[name] if p["iteration"] >= 500]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Iteração de treino")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k")
legend(axes[0], loc="upper right")
save(fig, "fig_lj_curves.png")
summary["lj_recon_final"] = {
    "baseline": round(ck["baseline_lj"][-1]["recon"], 3),
    "attn": round(ck["attn_lj"][-1]["recon"], 3),
    "baseline_best": round(min(p["recon"] for p in ck["baseline_lj"]), 3),
    "attn_best": round(min(p["recon"] for p in ck["attn_lj"]), 3),
}

# ================================================== 2. TB curves (attn_lj, verbo)
tb = json.load(open(HERE / "tb_curves.json"))
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
for ax, name, title in [(axes[0], "attn_lj", "LJSpeech c/ atenção — TensorBoard"),
                        (axes[1], "verbo_v4_f1", "VERBO f1 (v4) — TensorBoard")]:
    tr = [(s, v) for s, v in tb[name]["Loss/Train_Total"] if s >= 500]
    va = [(s, v) for s, v in tb[name]["Loss/Val_Total"] if s >= 500]
    ax.plot(*zip(*tr), color=BLUE, lw=2, label="Treino (total)")
    ax.plot(*zip(*va), color=ORANGE, lw=2, label="Validação (total)")
    ax.set_title(title)
    ax.set_xlabel("Iteração")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k")
legend(axes[0], loc="upper right")
legend(axes[1], loc="upper right")
save(fig, "fig_tb_curves.png")
summary["tb_final"] = {
    "attn_train": round(tb["attn_lj"]["Loss/Train_Total"][-1][1], 3),
    "attn_val": round(tb["attn_lj"]["Loss/Val_Total"][-1][1], 3),
    "verbo_train": round(tb["verbo_v4_f1"]["Loss/Train_Total"][-1][1], 3),
    "verbo_val": round(tb["verbo_v4_f1"]["Loss/Val_Total"][-1][1], 3),
}

# ================================================== 3. LJ attention overlays
lj_attn = json.load(open(ANA / "lj_attention.json"))
STYLE_LABEL = {"expressiva": "Expressiva (F0 muito variável)",
               "grave_lenta": "Grave e lenta",
               "rapida": "Rápida"}
fig, axes = plt.subplots(3, 1, figsize=(9.6, 7.2))
for ax, (style, d) in zip(axes, lj_attn.items()):
    mel = np.array(d["mel"])
    w = np.array(d["weights"])
    T = mel.shape[1]
    ax.imshow(mel, origin="lower", aspect="auto", cmap=MEL_CMAP,
              extent=[0, T * 256 / 22050, 0, 80], interpolation="none")
    ax.grid(False)
    # attention as step density: w*T' == 1 means uniform
    dens = w * len(w)
    edges = np.arange(len(w) + 1) * 64 * 256 / 22050
    ax.stairs(dens / dens.max() * 70 + 5, edges, color=ORANGE, lw=2.5,
              baseline=None)
    ax.set_title(f"{STYLE_LABEL[style]} — {d['utt']}", loc="left")
    ax.set_ylabel("Canal mel")
    ax.set_yticks([0, 40, 80])
axes[-1].set_xlabel("Tempo (s) — linha laranja: peso de atenção por janela de ~0,74 s")
fig.tight_layout()
save(fig, "fig_lj_attention.png")

# ================================================== 4-5. VERBO attention analyses
per_utt = json.loads((ANA / "verbo_attention_mu.json").read_text())
NBINS = 20


def resample_density(w, nbins=NBINS):
    """Attention density over normalized time (uniform attention == 1.0)."""
    w = np.asarray(w)
    dens = w * len(w)
    x_src = (np.arange(len(w)) + 0.5) / len(w)
    x_tgt = (np.arange(nbins) + 0.5) / nbins
    return np.interp(x_tgt, x_src, dens)


heat = np.zeros((len(EMO_ORDER), NBINS))
ent, corr = {}, {}
for i, e in enumerate(EMO_ORDER):
    utts = [u for u in per_utt if u["emotion"] == e]
    heat[i] = np.mean([resample_density(u["weights"]) for u in utts], axis=0)
    hs, cs = [], []
    for u in utts:
        w = np.asarray(u["weights"])
        if len(w) > 1:
            p = w / w.sum()
            hs.append(float(-(p * np.log(p + 1e-12)).sum() / np.log(len(w))))
            eb = np.asarray(u["energy_bins"])
            if np.std(eb) > 1e-6 and np.std(w) > 1e-9:
                cs.append(float(np.corrcoef(w, eb)[0, 1]))
    ent[e] = (float(np.mean(hs)), float(np.std(hs)))
    corr[e] = (float(np.mean(cs)), float(np.std(cs)))
summary["verbo_entropy"] = {k: [round(a, 3), round(b, 3)] for k, (a, b) in ent.items()}
summary["verbo_energy_corr"] = {k: [round(a, 3), round(b, 3)] for k, (a, b) in corr.items()}

fig, ax = plt.subplots(figsize=(8.4, 3.4))
im = ax.imshow(heat, aspect="auto", cmap=MEL_CMAP, vmin=heat.min(), vmax=heat.max())
ax.grid(False)
ax.set_yticks(range(len(EMO_ORDER)), EMO_ORDER)
ax.set_xticks([0, NBINS / 2 - 0.5, NBINS - 1], ["início", "meio", "fim"])
ax.set_xlabel("Tempo normalizado do áudio de referência")
ax.set_title("Densidade média de atenção por emoção (1,0 = atenção uniforme)", loc="left")
cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.015)
cb.outline.set_visible(False)
cb.ax.tick_params(color=MUTED, labelcolor=MUTED, labelsize=8)
save(fig, "fig_verbo_attn_heatmap.png")

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
xs = np.arange(len(EMO_ORDER))
for ax, data, title, ylab in [
        (axes[0], ent, "Entropia normalizada da atenção", "H / ln T′  (1,0 = uniforme)"),
        (axes[1], corr, "Correlação atenção × energia do mel", "Pearson r")]:
    vals = [data[e][0] for e in EMO_ORDER]
    errs = [data[e][1] for e in EMO_ORDER]
    ax.bar(xs, vals, 0.62, color=[EMO_COLORS[e] for e in EMO_ORDER], zorder=3)
    ax.errorbar(xs, vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1,
                capsize=2, zorder=4)
    for x, v in zip(xs, vals):
        ax.text(x, v + (0.02 if v >= 0 else -0.05), f"{v:.2f}", ha="center",
                fontsize=8, color=INK2)
    ax.set_xticks(xs, EMO_ORDER, rotation=30, ha="right")
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylab)
axes[1].axhline(0, color=BASELINE_AX, lw=1)
save(fig, "fig_verbo_attn_stats.png")

# example overlays for 3 emotions
ex = json.loads((ANA / "verbo_example_mels.json").read_text())
by_utt = {u["utt"]: u for u in per_utt}
fig, axes = plt.subplots(3, 1, figsize=(9.6, 7.2))
for ax, (e, d) in zip(axes, ex.items()):
    mel = np.array(d["mel"])
    w = np.array(by_utt[d["utt"]]["weights"])
    T = mel.shape[1]
    ax.imshow(mel, origin="lower", aspect="auto", cmap=MEL_CMAP,
              extent=[0, T * 256 / 22050, 0, 80], interpolation="none")
    ax.grid(False)
    dens = w * len(w)
    edges = np.arange(len(w) + 1) * 64 * 256 / 22050
    ax.stairs(dens / max(dens.max(), 1e-9) * 70 + 5, edges, color=ORANGE, lw=2.5)
    ax.set_title(f"{e} — {d['utt']} — “{d['text']}”", loc="left")
    ax.set_ylabel("Canal mel")
    ax.set_yticks([0, 40, 80])
axes[-1].set_xlabel("Tempo (s) — linha laranja: peso de atenção por janela de ~0,74 s")
fig.tight_layout()
save(fig, "fig_verbo_attention_examples.png")

# ================================================== 6. latent variance per dim
li = json.load(open(ANA / "latent_info.json"))
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2))
for ax, tag, title in [(axes[0], "base", "LJ baseline (32 dims)"),
                       (axes[1], "attn", "LJ c/ atenção (32 dims)"),
                       (axes[2], "verbo", "VERBO f1 (16 dims)")]:
    var = np.array(li[tag]["var_all"])
    top = [d["dim"] for d in li[tag]["top_dims"]]
    colors = [ORANGE if i in top else "#9ec5f4" for i in range(len(var))]
    ax.bar(range(len(var)), var, 0.7, color=colors, zorder=3)
    for d in top:
        ax.text(d, var[d], f" z{d}", ha="center", va="bottom", fontsize=8,
                color=INK2)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Dimensão do latente")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
axes[0].set_ylabel("Var(μ) no corpus")
axes[0].text(0.97, 0.72, "escala ~100×\nmenor: posterior\ncolapsado",
             transform=axes[0].transAxes, ha="right", fontsize=8.5, color=RED)
save(fig, "fig_latent_var.png")
summary["latent"] = {t: {"top_dims": li[t]["top_dims"],
                         "n_active_0p1": int((np.array(li[t]["var_all"]) > 0.1).sum()),
                         "max_var": float(max(li[t]["var_all"]))}
                     for t in ["base", "attn", "verbo"]}

# ================================================== 7. verbo latent PCA scatter
mus = np.array([u["mu"] for u in per_utt])
emos = [u["emotion"] for u in per_utt]
mu_c = mus - mus.mean(0)
U, S, Vt = np.linalg.svd(mu_c, full_matrices=False)
pcs = mu_c @ Vt[:2].T
evr = (S ** 2 / (S ** 2).sum())[:2]
fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
for ax, X, title, xl, yl in [
        (axes[0], pcs, f"PCA de μ ({100*evr.sum():.0f}% da variância)",
         f"PC1 ({100*evr[0]:.0f}%)", f"PC2 ({100*evr[1]:.0f}%)"),
        (axes[1], mus[:, [li["verbo"]["top_dims"][0]["dim"],
                          li["verbo"]["top_dims"][1]["dim"]]],
         "Dimensões mais influentes",
         f"z{li['verbo']['top_dims'][0]['dim']}",
         f"z{li['verbo']['top_dims'][1]['dim']}")]:
    for e in EMO_ORDER:
        idx = [i for i, x in enumerate(emos) if x == e]
        ax.scatter(X[idx, 0], X[idx, 1], s=34, color=EMO_COLORS[e], label=e,
                   edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
leg = axes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
for t in leg.get_texts():
    t.set_color(INK2)
save(fig, "fig_verbo_latent_scatter.png")
summary["verbo_pca_evr"] = [round(float(v), 3) for v in evr]

# ================================================== 8. interpolation norms
ii = json.load(open(ANA / "interp_info.json"))
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2))
ts = np.linspace(0, 1, 21)
for ax, tag, title in [(axes[0], "base", "LJ baseline"),
                       (axes[1], "attn", "LJ c/ atenção"),
                       (axes[2], "verbo", "VERBO f1")]:
    d = ii[tag]
    ax.plot(ts, d["norms_lerp"], color=RED, lw=2, label="lerp (linear)")
    ax.plot(ts, d["norms_slerp"], color=BLUE, lw=2, label="slerp (esférica)")
    ax.set_title(f"{title} — ângulo {d['angle_deg']:.0f}°", loc="left")
    ax.set_xlabel("t")
    ax.set_ylim(bottom=0)
axes[0].set_ylabel("‖z(t)‖")
legend(axes[2], loc="lower center")
save(fig, "fig_interp_norms.png")
summary["interp"] = {t: {"angle": round(ii[t]["angle_deg"], 1),
                         "norm_a": round(ii[t]["norm_a"], 2),
                         "norm_b": round(ii[t]["norm_b"], 2),
                         "lerp_min": round(min(ii[t]["norms_lerp"]), 2)}
                     for t in ["base", "attn", "verbo"]}

# ================================================== 9. interpolation mel strips
for tag, label in [("base", "LJ baseline"), ("attn", "LJ c/ atenção"),
                   ("verbo", "VERBO f1")]:
    z = np.load(ANA / f"interp_mels_{tag}.npz")
    fig, axes = plt.subplots(2, 5, figsize=(13, 4.2))
    for row, method in enumerate(["lerp", "slerp"]):
        for col, t in enumerate([0, 25, 50, 75, 100]):
            ax = axes[row][col]
            key = f"{method}_t{t:03d}"
            if key not in z.files:  # endpoints stored once under lerp
                key = f"lerp_t{t:03d}"
            mel = z[key]
            ax.imshow(mel, origin="lower", aspect="auto", cmap=MEL_CMAP,
                      interpolation="none")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t = {t/100:.2f}", fontsize=10)
            if col == 0:
                ax.set_ylabel(method, fontsize=11, color=INK)
    fig.suptitle(f"Mel-espectrogramas ao longo da interpolação — {label}",
                 fontsize=12, color=INK, x=0.5, y=1.0)
    fig.tight_layout()
    save(fig, f"fig_interp_mels_{tag}.png")

(HERE / "analysis_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
