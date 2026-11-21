#!/usr/bin/env python3
"""
How many bits about the reference voice reach the generated audio?

Reads embeddings.npz written by scripts/coverage/speaker_similarity.py in an eval folder and
measures, per reference group (seen / unseen / external), the information that the voice of
the output carries about which reference it came from, C = I(voice of output; reference):

    C_ce    lower bound from identification: each output is scored against every reference of
            the group by cosine similarity, turned into a softmax with inverse temperature beta
            (chosen on the other references, so no leakage); C_ce = log2(S) - cross-entropy
    C_fano  lower bound from identification accuracy (Fano's inequality)
    H_out, H_cond, C_gauss
            Gaussian entropies in the k leading principal directions of the outputs:
            H(Y) from the covariance of all outputs, H(Y|S) from the pooled covariance of the
            outputs of each reference (different sentences, same reference), C_gauss = H(Y) - H(Y|S).
            The split tells whether a gain in C comes from more distinct reachable voices
            (H(Y)) or from less noise per voice (H(Y|S)).

The cap is log2(S) for S references in the group. Intervals: bootstrap over references.
Identification uses the speaker centroid of each reference (ref_spk; the reference utterance
itself when no pool was given), and, as a ceiling, held-out real speech is not available here.

Example:
    python scripts/coverage/voice_bits.py --eval-dir /content/out/n8_k32_s1/step_2500/strength_1.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

BETAS = np.linspace(1, 150, 150)


def fano(acc: float, s: int) -> float:
    pe = min(max(1 - acc, 1e-12), 1 - 1e-12)
    h = -(pe * np.log2(pe) + (1 - pe) * np.log2(1 - pe))
    return float(np.log2(s) - h - pe * np.log2(s - 1))


def ce_rows(sim: np.ndarray, true: np.ndarray, by_ref: np.ndarray) -> np.ndarray:
    """Cross-entropy (bits) of each output, with beta chosen on the outputs of the other references."""
    logits = BETAS[:, None, None] * sim[None]                                     # (nb, n, S)
    ce = -(logits[:, np.arange(len(true)), true] - logsumexp(logits, axis=2)) / np.log(2)   # (nb, n)
    refs = np.unique(by_ref)
    tot = np.stack([ce[:, by_ref == r].sum(1) for r in refs], 1)                  # (nb, n_refs)
    out = np.empty(len(true))
    for j, r in enumerate(refs):
        k = np.argmin(tot.sum(1) - tot[:, j])
        out[by_ref == r] = ce[k, by_ref == r]
    return out


def gauss(Y: np.ndarray, lab: np.ndarray, P: np.ndarray) -> tuple:
    Z = (Y - Y.mean(0)) @ P.T
    st = np.cov(Z.T)
    sw = sum((Z[lab == r] - Z[lab == r].mean(0)).T @ (Z[lab == r] - Z[lab == r].mean(0)) for r in np.unique(lab))
    sw = sw / (len(Z) - len(np.unique(lab)))
    h = lambda c: 0.5 * np.log2(np.linalg.det(2 * np.pi * np.e * np.atleast_2d(c)))
    return h(st), h(sw), h(st) - h(sw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=6, help="Principal directions for the Gaussian entropies")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--sentences", type=int, default=0, help="Use only the first n test sentences (0 = all)")
    a = ap.parse_args()

    d = np.load(a.eval_dir / "embeddings.npz", allow_pickle=False)
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    out, ref_of = unit(d["output"]), d["reference"].astype(str)
    samples = d["sample"].astype(str)
    if a.sentences:
        keep = np.array([int(s.rsplit("__s", 1)[1]) < a.sentences for s in samples])
        out, ref_of = out[keep], ref_of[keep]
    ids, groups, cent = d["ref_id"].astype(str), d["ref_group"].astype(str), unit(d["ref_spk"])
    rng = np.random.default_rng(0)
    report = {}
    for g in sorted(set(groups)):
        gid = ids[groups == g]
        S = len(gid)
        if S < 2:
            continue
        idx = {r: i for i, r in enumerate(gid)}
        m = np.isin(ref_of, gid)
        Y, lab = out[m], np.array([idx[r] for r in ref_of[m]])
        sim = Y @ cent[groups == g].T
        correct = (sim.argmax(1) == lab).astype(float)
        rows = ce_rows(sim, lab, lab)
        k = min(a.k, S - 1)
        P = np.linalg.svd(Y - Y.mean(0), full_matrices=False)[2][:k]
        h_out, h_cond, c_g = gauss(Y, lab, P)
        bs = []
        for _ in range(a.bootstrap):
            pick = rng.integers(0, S, S)
            sel = np.concatenate([np.where(lab == j)[0] for j in pick])
            bs.append([fano(correct[sel].mean(), S), np.log2(S) - rows[sel].mean()])
        bs = np.array(bs)
        r = dict(S=S, cap=float(np.log2(S)), n_outputs=int(len(Y)), accuracy=float(correct.mean()),
                 C_ce=float(np.log2(S) - rows.mean()), C_ce_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
                 C_fano=fano(correct.mean(), S), C_fano_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                 k=k, H_out=float(h_out), H_cond=float(h_cond), C_gauss=float(c_g))
        report[g] = r
        print(f"{g:9s} S={S:3d} cap {r['cap']:.2f} | acc {r['accuracy']:.3f} | C_ce {r['C_ce']:.2f} "
              f"[{r['C_ce_ci'][0]:.2f}, {r['C_ce_ci'][1]:.2f}] | C_fano {r['C_fano']:.2f} | "
              f"H(Y) {h_out:.2f} H(Y|S) {h_cond:.2f} (k={k}) bits")
    (a.eval_dir / "voice_bits.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
