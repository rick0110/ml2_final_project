"""Grouped results table (markdown) from research/results/data/cluster_runs.csv.

One row per (variant, N, K, step): mean ± standard deviation over seeds of the voice bits
(cross-entropy bound, 38 external voices), the share of copied samples and the WER.

    python scripts/coverage/results_table.py --variant base nokl --step 10000
"""
import argparse, csv, statistics as st
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).resolve().parents[2] / "research/results/data/cluster_runs.csv"


def fmt(vals, d=2, pct=False):
    vals = [v * 100 if pct else v for v in vals]
    m = f"{st.mean(vals):.{d}f}"
    return m + (f" ± {st.pstdev(vals):.{d}f}" if len(vals) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--variant", nargs="*", help="e.g. base film nokl filmnokl (default: all)")
    ap.add_argument("--step", nargs="*", type=int)
    ap.add_argument("--N", nargs="*", type=int)
    ap.add_argument("--K", nargs="*", type=int)
    a = ap.parse_args()
    groups = defaultdict(list)
    for r in csv.DictReader(open(a.csv)):
        key = (r["variant"], int(r["N"]), int(r["K"]), int(r["step"]))
        if a.variant and key[0] not in a.variant: continue
        if a.N and key[1] not in a.N: continue
        if a.K and key[2] not in a.K: continue
        if a.step and key[3] not in a.step: continue
        groups[key].append(r)
    print("| Variant | N | K | Steps | Seeds | Voice bits | Copied (%) | WER (%) |")
    print("|---|---|---|---|---|---|---|---|")
    for k in sorted(groups, key=lambda k: (k[0] != "base", k[0], k[3], k[2], k[1])):
        g = groups[k]
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {len(g)} | {fmt([float(x['bits_ce']) for x in g])} | "
              f"{fmt([float(x['copied']) for x in g], d=0, pct=True)} | {fmt([float(x['wer']) for x in g], d=1, pct=True)} |")


if __name__ == "__main__":
    main()
