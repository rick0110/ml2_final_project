#!/usr/bin/env python3
"""
What does the style latent encode? Analysis of latent means saved by scripts/eval/latent_dump.py.

For a set of labelled utterances (speaker, emotion, sentence) it reports:
  - linear probes: accuracy of a logistic regression that predicts each label from mu
    (5-fold cross-validation; for emotion also leaving whole speakers out);
  - variance explained: share of the latent variance that lies between the groups of each label;
  - geometry: distance from each speaker's centroid to the centroid of a reference corpus
    (the voice with wide text coverage), in units of that corpus' spread;
  - a 2-D PCA projection.

Example:
    python scripts/coverage/latent_analysis.py --latents latents/mix.npz \
        --labelled VERBO --anchor TTS-Portuguese --out analysis/mix
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def probe(x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> float:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = GroupKFold(n_splits=min(5, len(set(groups)))) if groups is not None else StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(clf, x, y, cv=cv, groups=groups).mean())


def between_share(x: np.ndarray, y: np.ndarray) -> float:
    """Between-group sum of squares over total sum of squares, summed over latent dimensions."""
    total = ((x - x.mean(0)) ** 2).sum()
    between = sum((y == g).sum() * ((x[y == g].mean(0) - x.mean(0)) ** 2).sum() for g in set(y))
    return float(between / total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latents", type=Path, required=True)
    ap.add_argument("--labelled", required=True, help="`source` value of the labelled corpus (e.g. VERBO)")
    ap.add_argument("--anchor", default=None, help="`source` value of the wide-coverage corpus")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    d = np.load(a.latents)
    lab = np.array([a.labelled in s for s in d["source"]])
    x = d["mu"][lab]
    speaker, emotion, text = d["speaker"][lab], d["emotion"][lab], d["text"][lab]
    report: Dict[str, object] = {"n": int(lab.sum()), "latent_dim": int(x.shape[1])}

    labels = {"speaker": speaker, "sentence": text}
    if len(set(emotion)) > 1:
        labels["emotion"] = emotion
    gender = np.array([re.sub(r"\d", "", s) for s in speaker])
    if set(gender) == {"f", "m"}:
        labels["gender"] = gender
    for name, y in labels.items():
        counts = np.unique(y, return_counts=True)[1]
        report[name] = {"classes": len(counts), "chance": round(float(counts.max() / counts.sum()), 3),
                        "probe_accuracy": round(probe(x, y), 3), "variance_between": round(between_share(x, y), 3)}
    if "emotion" in labels:
        report["emotion"]["probe_accuracy_unseen_speakers"] = round(probe(x, emotion, groups=speaker), 3)
        # emotion inside each speaker: remove the speaker centroid first
        centred = x.copy()
        for s in set(speaker):
            centred[speaker == s] -= x[speaker == s].mean(0)
        report["emotion"]["variance_between_within_speaker"] = round(between_share(centred, emotion), 3)
        report["sentence"]["variance_between_within_speaker"] = round(between_share(centred, text), 3)

    pca = PCA(2).fit(d["mu"])
    proj = pca.transform(d["mu"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    if a.anchor:
        anc = np.array([a.anchor in s for s in d["source"]])
        xa = d["mu"][anc]
        centre, spread = xa.mean(0), xa.std(0).mean()
        report["anchor"] = {"n": int(anc.sum()), "spread": round(float(spread), 4)}
        report["distance_to_anchor"] = {s: round(float(np.linalg.norm(x[speaker == s].mean(0) - centre) / spread), 2)
                                        for s in sorted(set(speaker))}
        report["norm_ratio_labelled_over_anchor"] = round(float(np.linalg.norm(x, axis=1).mean() / np.linalg.norm(xa, axis=1).mean()), 2)
        for ax in axes:
            ax.scatter(*proj[anc].T, s=5, color="#9ca3af", alpha=0.5, label=a.anchor)
    for ax, (name, y) in zip(axes, [("speaker", speaker), ("emotion" if "emotion" in labels else "sentence", labels.get("emotion", text))]):
        cmap = plt.get_cmap("tab20" if len(set(y)) > 10 else "tab10")
        for i, g in enumerate(sorted(set(y))):
            ax.scatter(*proj[lab][y == g].T, s=7, color=cmap(i), label=str(g)[:18])
        ax.set_title(f"cor = {name}", fontsize=10)
        ax.legend(fontsize=6, ncol=2, frameon=False, markerscale=1.5)
        ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.0f}% da variância)")
        ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.0f}%)")
    fig.tight_layout()
    fig.savefig(a.out / "pca.png", dpi=120)
    plt.close(fig)

    (a.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
