#!/usr/bin/env python3
"""
Figures of the retraining (report and project page), built from the training logs
(metrics.jsonl written by train.py) and the evaluation tables (results.csv written by
scripts/eval/synth_check.py).

Example:
    python scripts/site_assets/make_figures.py --runs runs --evals eval --out docs/assets/img

    runs/<experiment>/metrics.jsonl      evals/<experiment>/results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STAGES = [("en_vae_v1", "Inglês (LJSpeech)"), ("pt_vae_v1", "Português (TTS-Portuguese)"),
          ("ptverbo_vae_v1", "Português + VERBO")]
BLUE, ORANGE, GRAY = "#2563eb", "#ea580c", "#6b7280"
EMO_COLORS = {"neutro": "#6b7280", "alegria": "#eab308", "tristeza": "#2563eb", "raiva": "#dc2626",
              "surpresa": "#9333ea", "medo": "#0d9488", "desgosto": "#65a30d"}


def read_metrics(path: Path) -> List[dict]:
    """Last record wins when a step was logged twice (a preempted VM resumes from a checkpoint)."""
    by_key: Dict[tuple, dict] = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        by_key[(r["it"], bool(r.get("eval")))] = r
    return [by_key[k] for k in sorted(by_key)]


def read_eval(path: Path) -> List[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8")))


def fig_curves(runs: Path, out: Path) -> None:
    stages = [(e, t) for e, t in STAGES if (runs / e / "metrics.jsonl").exists()]
    fig, axes = plt.subplots(2, len(stages), figsize=(4.2 * len(stages), 5.2), squeeze=False)
    for j, (exp, title) in enumerate(stages):
        rec = read_metrics(runs / exp / "metrics.jsonl")
        tr = [r for r in rec if not r.get("eval")]
        ev = [r for r in rec if r.get("eval")]
        ax = axes[0][j]
        ax.plot([r["it"] for r in tr], [r["recon"] for r in tr], color=GRAY, lw=0.8, label="treino")
        ax.plot([r["it"] for r in ev], [r["val_recon"] for r in ev], color=BLUE, marker="o", ms=3, label="validação")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("erro de reconstrução do mel")
        ax.set_ylim(0.4, min(2.2, max([r["val_recon"] for r in ev] + [r["recon"] for r in tr[2:]]) * 1.05))
        ax.legend(frameon=False, fontsize=8)
        ax = axes[1][j]
        ax.plot([r["it"] for r in ev], [r["align_peak"] for r in ev], color=ORANGE, marker="o", ms=3)
        ax.set_ylim(0, 1)
        ax.set_ylabel("nitidez do alinhamento (síntese livre)")
        ax.set_xlabel("passo de treino")
        ax2 = ax.twinx()
        ax2.plot([r["it"] for r in ev], [r["active_dims"] for r in ev], color=BLUE, ls="--", lw=1)
        ax2.set_ylim(0, 33)
        ax2.set_ylabel("dimensões ativas do latente (de 32)", color=BLUE)
    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_retreino_curvas.png", dpi=130)
    plt.close(fig)


def fig_wer(evals: Path, out: Path, old_en: float, old_pt: float) -> None:
    bars = [("Inglês\nmodelo antigo", old_en, GRAY), ("Português\nmodelo antigo", old_pt, GRAY)]
    for exp, label in [("en_vae_v1", "Inglês\nretreinado"), ("pt_vae_v1", "Português\nretreinado"),
                       ("verbo_vae_v1", "Emoção\nsó VERBO"), ("ptverbo_vae_v1", "Emoção\nPT + VERBO"),
                       ("mix_strength_0.6", "Emoção, PT + VERBO\nestilo a 60%")]:
        p = evals / exp / "results.csv"
        if p.exists():
            bars.append((label, float(np.mean([float(r["wer"]) for r in read_eval(p)])), BLUE))
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.bar(range(len(bars)), [100 * b[1] for b in bars], color=[b[2] for b in bars], width=0.62)
    for i, b in enumerate(bars):
        ax.text(i, 100 * b[1] + 2, f"{100 * b[1]:.0f}%".replace(".", ","), ha="center", fontsize=9)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel("palavras erradas (%)")
    ax.set_ylim(0, 115)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_retreino_wer.png", dpi=130)
    plt.close(fig)


def fig_style(evals: Path, out: Path, exp: str) -> None:
    """Does the generated speech follow the reference? Mean F0 and speaking rate, one point per reference."""
    p = evals / exp / "results.csv"
    if not p.exists():
        return
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in read_eval(p):
        groups[r["reference"]].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for (col, ref_col, name, unit), ax in zip(
            [("f0_hz", "ref_f0_hz", "Tom médio (F0)", "Hz"), ("chars_per_s", "ref_chars_per_s", "Velocidade", "caracteres/s")], axes):
        xs, ys = [], []
        for ref, rows in groups.items():
            x = float(rows[0][ref_col])
            y = float(np.nanmean([float(r[col]) for r in rows]))
            emo = rows[0]["reference_emotion"]
            ax.scatter(x, y, s=60, color=EMO_COLORS.get(emo, BLUE), marker="s" if "-m" in ref else "o", zorder=3)
            ax.annotate(emo or ref, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7)
            xs.append(x)
            ys.append(y)
        lo, hi = min(xs + ys) * 0.9, max(xs + ys) * 1.08
        ax.plot([lo, hi], [lo, hi], color=GRAY, ls=":", lw=1)
        r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")
        ax.set_title(f"{name} — correlação {r:.2f}".replace(".", ","), fontsize=10)
        ax.set_xlabel(f"áudio de referência ({unit})")
        ax.set_ylabel(f"fala gerada ({unit})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / f"fig_retreino_estilo_{exp}.png", dpi=130)
    plt.close(fig)


def fig_strength(evals: Path, out: Path, full: str, pattern: str = "mix_strength_*") -> None:
    """Trade-off of the style-strength control: word error rate and F0 spread across emotion references."""
    runs = {1.0: evals / full / "results.csv"}
    for d in sorted(evals.glob(pattern)):
        runs[float(d.name.rsplit("_", 1)[1])] = d / "results.csv"
    if len(runs) < 2 or not runs[1.0].exists():
        return
    refs = {r["reference"] for r in read_eval(runs[min(runs)])}
    xs, wers, lo, hi = [], [], [], []
    for a in sorted(runs):
        groups: Dict[str, List[dict]] = defaultdict(list)
        for r in read_eval(runs[a]):
            if r["reference"] in refs:
                groups[r["reference"]].append(r)
        f0 = [float(np.nanmean([float(r["f0_hz"]) for r in rows])) for rows in groups.values()]
        xs.append(a)
        wers.append(100 * float(np.mean([np.mean([float(r["wer"]) for r in rows]) for rows in groups.values()])))
        lo.append(min(f0))
        hi.append(max(f0))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    axes[0].plot(xs, wers, color=BLUE, marker="o")
    for x, w in zip(xs, wers):
        axes[0].annotate(f"{w:.0f}%", (x, w), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    axes[0].set_ylabel("palavras erradas (%)")
    axes[0].set_ylim(0, max(wers) * 1.25)
    axes[1].fill_between(xs, lo, hi, color=ORANGE, alpha=0.25)
    axes[1].plot(xs, lo, color=ORANGE, marker="o")
    axes[1].plot(xs, hi, color=ORANGE, marker="o")
    axes[1].set_ylabel("tom médio gerado: menor e maior\nentre as 7 emoções (Hz)")
    for ax in axes:
        ax.set_xlabel("intensidade do estilo (1 = referência inteira)")
        ax.set_xticks(xs)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_retreino_intensidade.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--evals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--old-wer-en", type=float, default=0.68, help="WER of the audio shipped before the retraining")
    ap.add_argument("--old-wer-pt", type=float, default=1.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig_curves(a.runs, a.out)
    fig_wer(a.evals, a.out, a.old_wer_en, a.old_wer_pt)
    for exp in ["pt_vae_v1", "verbo_vae_v1", "ptverbo_vae_v1"]:
        fig_style(a.evals, a.out, exp)
    fig_strength(a.evals, a.out, "ptverbo_vae_v1")
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
