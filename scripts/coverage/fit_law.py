#!/usr/bin/env python3
"""
Fit the two-level sampling law of reports/teoria_cobertura.md to the coverage runs.

Share of samples where the voice was NOT copied, at step 2500 and full style strength, in runs
without a variant suffix (n<N>_k<K>_s<seed>):
    unseen voices:  a + A / N + B / (N K)
    seen voices:    a + C / K + A / N
The simpler one-term alternatives are fitted too, for comparison.

Example:
    python scripts/coverage/fit_law.py --results reports/coverage/results
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

LAWS = {
    "unseen": [("a + A/N + B/(NK)", lambda p, n, k: p[0] + p[1] / n + p[2] / (n * k), 3),
               ("a + A/N", lambda p, n, k: p[0] + p[1] / n, 2),
               ("a + B/(NK)", lambda p, n, k: p[0] + p[1] / (n * k), 2),
               ("a + C/K", lambda p, n, k: p[0] + p[1] / k, 2)],
    "seen": [("a + C/K + A/N", lambda p, n, k: p[0] + p[1] / k + p[2] / n, 3),
             ("a + C/K", lambda p, n, k: p[0] + p[1] / k, 2),
             ("a + A/N", lambda p, n, k: p[0] + p[1] / n, 2),
             ("a + B/(NK)", lambda p, n, k: p[0] + p[1] / (n * k), 2)],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--step", type=int, default=2500)
    ap.add_argument("--strength", default="1.0")
    a = ap.parse_args()

    points = {"seen": [], "unseen": []}
    for path in sorted(a.results.glob(f"*/step_{a.step}/strength_{a.strength}/results.csv")):
        m = re.fullmatch(r"n(\d+)_k(\d+)_s(\d+)", path.parts[-4])
        if not m:
            continue
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for kind in points:
            sel = [r for r in rows if r["reference_emotion"] == kind]
            missed = 1 - np.mean([float(r["sim_ref"]) > float(r["sim_default"]) for r in sel])
            points[kind].append((int(m.group(1)), int(m.group(2)), missed))

    for kind, laws in LAWS.items():
        x = np.array(points[kind], dtype=float)
        print(f"{kind}: {len(x)} runs")
        for name, law, size in laws:
            fit = least_squares(lambda p: law(p, x[:, 0], x[:, 1]) - x[:, 2], [0.3] + [1.0] * (size - 1), bounds=(0, np.inf))
            print(f"  {name:18s} parameters {np.round(fit.x, 2)}  rmse {np.sqrt(np.mean(fit.fun ** 2)):.3f}")


if __name__ == "__main__":
    main()
