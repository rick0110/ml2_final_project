#!/usr/bin/env python3
"""
Table and figure of the coverage experiment (N speakers x K sentences per speaker).

Reads the results.csv of every run under --results (as written by scripts/coverage/colab_job.sh:
<run>/step_<n>/strength_<s>/results.csv, with run = n<N>_k<K>_s<seed>) and reports, for
references of trained ("seen") and never-trained ("unseen") voices:
    wer      word error rate of the generated audio
    copied   share of samples closer to the reference voice than to the default voice
             (sim_ref > sim_default, see scripts/coverage/speaker_similarity.py)

Example:
    python scripts/coverage/summarize.py --results reports/coverage/results \
        --figure reports/figures/fig_cobertura.png
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {2: "#2a78d6", 8: "#eb6834", 32: "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#1a1a19", "#6b6a63", "#e6e5df", "#fcfcfb"


def read(results: Path, step: int, strength: str) -> Dict[Tuple[int, int, str, str], Dict[str, List[float]]]:
    """(N, K, variant, kind) -> per-seed means of wer and copied."""
    out: Dict[Tuple[int, int, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(results.glob(f"*/step_{step}/strength_{strength}/results.csv")):
        m = re.fullmatch(r"n(\d+)_k(\d+)_s(\d+)(-\w+)?", path.parts[-4])
        if not m:
            continue
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for kind in ("seen", "unseen"):
            sel = [r for r in rows if r["reference_emotion"] == kind]
            key = (int(m.group(1)), int(m.group(2)), (m.group(4) or "")[1:], kind)
            out[key]["wer"].append(float(np.mean([float(r["wer"]) for r in sel])))
            out[key]["copied"].append(float(np.mean([float(r["sim_ref"]) > float(r["sim_default"]) for r in sel])))
    return out


def cell(values: List[float]) -> str:
    return " / ".join(f"{100 * v:.0f}%" for v in values)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("reports/coverage/results"))
    ap.add_argument("--step", type=int, default=2500)
    ap.add_argument("--strength", default="1.0")
    ap.add_argument("--figure", type=Path, default=None)
    a = ap.parse_args()

    data = read(a.results, a.step, a.strength)
    print("| N | K | variant | seeds | WER seen | WER unseen | copied seen | copied unseen |")
    print("|---|---|---|---|---|---|---|---|")
    for n, k, variant in sorted({key[:3] for key in data}, key=lambda t: (t[2], t[0], t[1])):
        seen, unseen = data[(n, k, variant, "seen")], data[(n, k, variant, "unseen")]
        print(f"| {n} | {k} | {variant or '-'} | {len(seen['wer'])} | {cell(seen['wer'])} | {cell(unseen['wer'])} "
              f"| {cell(seen['copied'])} | {cell(unseen['copied'])} |")

    if a.figure is None:
        return
    ks = sorted({key[1] for key in data})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, facecolor=SURFACE)
    for ax, kind, title in zip(axes, ("seen", "unseen"), ("Vozes treinadas", "Vozes nunca vistas")):
        ax.set_facecolor(SURFACE)
        labels: List[list] = []
        for n in sorted({key[0] for key in data}):
            xs, means = [], []
            for k in ks:
                seeds = data.get((n, k, "", kind), {}).get("copied", [])
                if not seeds:
                    continue
                xs.append(ks.index(k))
                means.append(100 * float(np.mean(seeds)))
                ax.scatter([ks.index(k)] * len(seeds), [100 * s for s in seeds], s=22, color=COLORS[n],
                           edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.plot(xs, means, color=COLORS[n], linewidth=2, zorder=2)
            labels.append([means[-1], xs[-1], f"{n} locutores"])
        labels.sort()
        for lower, upper in zip(labels, labels[1:]):  # keep the line labels apart
            upper[0] = max(upper[0], lower[0] + 7)
        for y, x, text in labels:
            ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_xlim(-0.3, len(ks) - 1 + 0.95)
        ax.set_ylim(0, 100)
        ax.set_xlabel("frases por locutor no treino", fontsize=9, color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("voz copiada (% das amostras)", fontsize=9, color=MUTED)
    fig.tight_layout()
    a.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.figure, dpi=200, facecolor=SURFACE)
    print(f"figure: {a.figure}")


if __name__ == "__main__":
    main()
